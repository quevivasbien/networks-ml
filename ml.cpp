#include "ml.h"


void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

void write_to_csv(
    const torch::Tensor& tensor,
    const std::string& filename
) {
    assert(tensor.dim() == 2);  // expects a 2d tensor
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < tensor.size(0); i++) {
            for (int j = 0; j < tensor.size(1); j++) {
                file << tensor.index({i, j}).item<float>() << ',';
            }
            file << '\n';
        }
        file.close();
    }
    else {
        std::cout << "Unable to open file.\n";
    }
}


MLModel::MLModel(
    int graph_size,
    int max_signals,
    int n_hidden,
    int hidden_size
) : graph_size(graph_size)
{
    // input features are:
    // graph_size**2 features to describe the graph
    // graph_size to indicate guesser id
    // for each graph member * for each of max_signals:
    // graph_size features to indicate signal sender id
    // 1 for signal value
    n_other_features = graph_size * max_signals * (graph_size + 1);
    n_features = graph_size * (graph_size + 1) + n_other_features;

    first = register_module(
        "first",
        torch::nn::Linear(n_features, hidden_size)
    );
    for (int i = 0; i < n_hidden; i++) {
        hidden.push_back(
            register_module(
                "hidden_" + std::to_string(i),
                torch::nn::Linear(hidden_size, hidden_size)
            )
        );
    }
    last = register_module(
        "last",
        torch::nn::Linear(hidden_size, graph_size)
    );

    this->apply(xavier_init);
}

torch::Tensor MLModel::get_features(const Network& net) {
    assert(net.graph().size() == graph_size);
    auto graph_features = torch::tile(
        torch::tensor(as_type<float, bool>(net.graph().values())),
        {graph_size, 1}
    );
    auto id_features = torch::eye(graph_size);
    auto other_features = torch::zeros({graph_size, n_other_features});
    // iterate through network members
    auto net_members = net.net_members();
    for (int i = 0; i < net_members.size(); i++) {
        // iterate through signal sources
        auto signals_set = net_members[i].signals();
        for (int j = 0; j < signals_set.size(); j++) {
            const auto& signals = signals_set[j];
            int start_idx = max_signals * (graph_size + 1) * j;
            std::cout << "start_idx: " << start_idx << '\n';
            // iterate through signal senders
            for (
                int k = 0;
                k < signals.size() && k < max_signals;
                k++
            ) {
                const Signal& signal = signals[k];
                int offset = start_idx + (graph_size + 1) * k;
                std::cout << offset << ' ' << signal.sender << ' ' << i << ' ' << graph_size << '\n';
                other_features.index({i, offset + signal.sender}) = 1.0;
                other_features.index({i, offset + graph_size}) = signal.value;
            }
        }
    }
    return torch::concat({graph_features, id_features, other_features}, 1);
}

torch::Tensor MLModel::forward(torch::Tensor x) {
    x = torch::relu(first->forward(x));
    for (auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    return last->forward(x);
}

void MLModel::get_nets_features_targets_helper(
    std::vector<Network>* nets,
    std::vector<torch::Tensor>* features,
    std::vector<torch::Tensor>* targets
) {
    auto graph = gen_random_graph(graph_size, connection_proba);
    Network new_net(graph, value_mean, value_sd, signal_life, signal_error_sd);
    auto new_features = get_features(new_net);
    auto new_targets = torch::tile(
        torch::tensor(new_net.values()),
        {graph_size, 1}
    );
    {
        std::lock_guard<std::mutex> lock(myMutex);
        nets->push_back(new_net);
        features->push_back(new_features);
        targets->push_back(new_targets);
    }
}

std::tuple<
    std::vector<Network>, torch::Tensor, torch::Tensor
> MLModel::get_nets_features_targets(int n, int threads) {
    std::vector<Network> networks;
    networks.reserve(n);
    std::vector<torch::Tensor> features;
    features.reserve(n);
    std::vector<torch::Tensor> targets;
    targets.reserve(n);
    while (networks.size() < n) {
        int pool_size = (threads < n) ? threads : n;
        std::vector<std::thread> threads;
        threads.reserve(pool_size);
        for (int i = 0; i < pool_size; i++) {
            threads.push_back(
                std::thread(
                    &MLModel::get_nets_features_targets_helper,
                    this,
                    &networks,
                    &features,
                    &targets
                )
            );
        }
        for (int i = 0; i < pool_size; i++) {
            threads[i].join();
        }
    }
    return std::make_tuple(
        networks,
        torch::concat(features),
        torch::concat(targets)
    );
}

std::vector<double> MLModel::train(
    int epochs,
    double lr,
    int epoch_size
) {
    auto optimizer = torch::optim::Adam(parameters(), lr);
    auto loss_fn = torch::nn::MSELoss();
    std::vector<double> loss_history;
    loss_history.reserve(epochs);
    for (int i = 0; i < epochs; i++) {
        auto [_, features, targets] = get_nets_features_targets(epoch_size, threads);
        optimizer.zero_grad();
        auto loss = loss_fn(forward(features), targets);
        loss_history.push_back(loss.item<double>());
        loss.backward();
        optimizer.step();
        if ((i + 1) % 10 == 0 || i + 1 == epochs) {
            std::cout << "Epoch " << i + 1 << " of " << epochs << ": Loss = " << loss.item<double>() << '\n';
        }
    }
    return loss_history;
}

torch::Tensor MLModel::eval_single(
    const Network& network,
    const torch::Tensor& features,
    const torch::Tensor& targets
) {
    torch::NoGradGuard noGrad;
    auto scores = (forward(features) - targets).pow(2);
    auto distances = torch::tensor(
        network.get_distances().values()
    ).view({graph_size, graph_size});
    return torch::concat({distances, scores}, 1);
}

void MLModel::eval(
    int n,
    std::string out_file
) {
    auto [networks, features, targets] = get_nets_features_targets(n, threads);
    std::vector<torch::Tensor> batch_data;
    batch_data.reserve(n);
    for (int i = 0; i < n; i++) {
        batch_data.push_back(
            eval_single(
                networks[i],
                features.index(
                    {torch::indexing::Slice(i*graph_size, (i+1)*graph_size)}
                ),
                targets.index(
                    {torch::indexing::Slice(i*graph_size, (i+1)*graph_size)}
                )
            )
        );
    }
    auto all_data = torch::concat(batch_data);
    write_to_csv(all_data, out_file);
}