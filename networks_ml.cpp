#include <torch/torch.h>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <random>
#include <assert.h>

#include <iostream>


std::default_random_engine rng;


template <typename T>
class Matrix {
public:
    Matrix(std::vector<T> values, std::size_t size) : values_(values), size_(size) {
        assert(values_.size() == size * size);
    }

    std::size_t size() const {
        return size_;
    }

    T get(std::size_t i, std::size_t j) const {
        assert(i < size_ && j < size_);
        return values_[i * size_ + j];
    }

    void set(std::size_t i, std::size_t j, T new_val) {
        assert(i < size_ && j < size_);
        values_[i * size_ + j] = new_val;
    }

    std::vector<T> values() const {
        return values;
    }

    void print() const {
        for (std::size_t i = 0; i < size_; i++) {
            for (std::size_t j = 0; j < size_; j++) {
                std::cout << (get(i, j)) ? '1' : '0';
                if (j == size_ - 1) {
                    std::cout << '\n';
                }
                else {
                    std::cout << ' ';
                }
            }
        }
    }

private:
    std::vector<T> values_;
    std::size_t size_;
};

Matrix<bool> gen_random_graph(std::size_t size, double p) {
    assert(0.0 <= p <= 1.0);
    std::vector<bool> values;
    values.reserve(size * size);
    std::uniform_real_distribution<double> dist(0, 1);
    for (std::size_t i = 0; i < size * (size - 1) / 2; i++) {
        values.push_back(dist(rng));
    }
    return Matrix<bool>(values, size);
}

std::vector<double> sample_normal(double mean, double sd, std::size_t size) {
    std::vector<double> out;
    std::normal_distribution<double> dist(mean, sd);
    out.reserve(size);
    for (std::size_t i = 0; i < size; i++) {
        out.push_back(dist(rng));
    }
    return out;
}


struct Signal {
    Signal(
        std::size_t sender, std::size_t source, double value
    ) : sender(sender), source(source), value(value) {}
    std::size_t sender;
    std::size_t source;
    double value;
};


class NetMember {
public:
    NetMember(
        std::size_t graph_size, std::size_t my_idx, double my_value
    ) : graph_size(graph_size), my_idx(my_idx), to_share({Signal(my_idx, my_idx, my_value)}) {
        signals.resize(graph_size);
    }

    void receive_signal(Signal signal) {
        if (is_new_source_and_origin(signal)) {
            signals[signal.source].push_back(signal);
            recently_received.push_back(signal);
        }
    }

    void flush_recent_signals() {
        to_share = recently_received;
        recently_received = {};
    }

    std::size_t get_my_idx() const {
        return my_idx;
    }

    const std::vector<Signal>& get_signals_to_share() const {
        return to_share;
    }

    void print() const {
        std::cout << "Signals received by member " << my_idx << ":\n";
        for (const auto& signal_set : signals) {
            for (const Signal& signal : signal_set) {
                std::cout << "Source: " << signal.source
                    << ", Sender: " << signal.sender
                    << ", Value: " << signal.value << '\n';
            }
        }
        std::cout << '\n';
    }

private:
    bool is_new_source_and_origin(const Signal& signal) const {
        if (signal.source == my_idx) {
            // This signal is about me, so I don't need it
            return false;
        }
        for (const Signal& my_signal : signals[signal.source]) {
            if (my_signal.sender == signal.sender) {
                // I already have a signal about this source from this sender
                return false;
            }
        }
        return true;
    }

    std::size_t graph_size;
    std::size_t my_idx;

    std::vector<std::vector<Signal>> signals;
    std::vector<Signal> to_share;
    std::vector<Signal> recently_received;
};


class Network {
public:
    Network(
        Matrix<bool> graph,
        double value_mean,
        double value_sd,
        unsigned int signal_life,
        double signal_error_sd
    ) : graph_(graph),
        signal_life(signal_life),
        signal_error_sd(signal_error_sd),
        values_(sample_normal(value_mean, value_sd, graph.size())) 
    {
        set_signals();
    }

    Matrix<int> get_distances() const {
        std::size_t size = graph_.size();
        // organize index of neighbors & create a distances matrix based on graph
        std::vector<int> values;
        values.reserve(size * size);
        std::vector<std::vector<std::size_t>> neighbors(size);
        for (std::size_t i = 0; i < size; i++) {
            for (std::size_t j = 0; j < size; j++) {
                bool graph_val = graph_.get(i, j);
                int distance = (i == j) ? 0 : ((graph_val) ? 1 : -1);
                values.push_back(distance);
                if (graph_.get(i, j)) {
                    neighbors[i].push_back(j);
                }
            }
        }
        Matrix<int> distances(values, size);

        unsigned int n = 1;
        bool keep_going = true;
        while (keep_going) {
            keep_going = false;
            for (std::size_t i = 0; i < size; i++) {
                for (std::size_t j = 0; j < size; j++) {
                    if (i == j || distances.get(i, j) != n) {
                        continue;
                    }
                    keep_going = true;
                    for (auto k : neighbors[j]) {
                        if (i != k && distances.get(i, k) == -1) {
                            distances.set(i, k, n + 1);
                        }
                    }
                }
            }
            n++;
        }
        return distances;
    }

    void print_signals() {
        for (const NetMember& member : net_members_) {
            member.print();
        }
    }

    const Matrix<bool>& graph() const {
        return graph_;
    }

    const std::vector<double>& values() const {
        return values_;
    }

    const std::vector<NetMember>& net_members() const {
        return net_members_;
    }

private:
    void set_signals() {
        // get things set up
        std::size_t size = graph_.size();
        std::vector<NetMember> members;
        members.reserve(size);
        for (std::size_t i = 0; i < size; i++) {
            members.push_back(NetMember(size, i, values_[i]));
        }
        std::normal_distribution<double> noise_dist(0.0, signal_error_sd);

        // now fill in signals
        // loop through time periods
        for (unsigned int t = 0; t < signal_life; t++) {
            // loop through signal senders
            for (std::size_t i = 0; i < size; i++) {
                const auto& signals_to_share = members[i].get_signals_to_share();
                // loop through signal receivers
                for (std::size_t j = 0; j < size; j++) {
                    if (i == j || !graph_.get(i, j)) {
                        continue;
                    }
                    for (const Signal& s : signals_to_share) {
                        members[j].receive_signal(Signal(i, s.source, s.value + noise_dist(rng)));
                    }
                }
                members[i].flush_recent_signals();
            }
        }

        net_members_ = members;
    }

    Matrix<bool> graph_;
    unsigned int signal_life;
    double signal_error_sd;
    std::vector<double> values_;
    std::vector<NetMember> net_members_;
};



void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}


struct MLModel : torch::nn::Module {
    MLModel(
        // graph params
        std::size_t graph_size,
        double connection_proba = 0.5,
        double value_mean = 0.0,
        double value_sd = 1.0,
        std::size_t signal_life = 5,
        double signal_error_sd = 0.1,
        // model params
        std::size_t max_signals = 4,
        std::size_t n_hidden = 4,
        std::size_t hidden_size = 50
    ) : graph_size(graph_size),
        max_signals(max_signals),
        connection_proba(connection_proba),
        value_mean(value_mean),
        value_sd(value_sd),
        signal_life(signal_life),
        signal_error_sd(signal_error_sd)
    {
        // input features are:
        // graph_size**2 features to describe the graph
        // graph_size to indicate guesser id
        // for each graph member * for each of max_signals:
        // graph_size features to indicate signal sender id
        // 1 for signal value
        n_other_features = graph_size + max_signals + (graph_size + 1);
        n_features = graph_size * (graph_size + 1) + n_other_features;

        first = register_module(
            "first",
            torch::nn::Linear(n_features, hidden_size)
        );
        for (std::size_t i = 0; i < n_hidden; i++) {
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

    torch::Tensor get_features(const Network& net) {
        assert(net.graph().size() == graph_size);
        auto graph_features = torch::tile(
            torch::from_blob(
                net.graph().values(), torch::dtype(torch::kFloat32)
            ),
            {graph_size, n_other_features}
        );
        auto id_features = torch::eye(graph_size);
        auto other_features = torch::zeros({graph_size, n_other_features});
        // iterate through network members
        auto net_members = net.net_members();
        for (std::size_t i = 0; i < net_members.size(); i++) {
            // iterate through signal sources
            auto signals_set = net_members[i].signals;
            for (std::size_t j = 0; j < signals_set.size(); j++) {
                const auto& signals = signals_set[j];
                std::size_t start_idx = max_signals * (graph_size + 1) * j;
                // iterate through signal senders
                for (
                    std::size_t k = 0;
                    k < signals.size() && k < max_signals;
                    k++
                ) {
                    const Signal& signal = signals[k];
                    std::size_t offset = start_idx + (graph_size + 1) * k;
                    other_features.index({i, offset + signal.sender}) = 1.0;
                    other_features.index({i, offset + graph_size}) = signal.value
                }
            }
        }
        return torch::concat({graph_features, id_features, other_features}, 1);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(first->forward(x));
        for (auto& h : hidden) {
            x = x + torch::relu(h->forward(x));
        }
        return last->forward(x);
    }

    std::vector<double> train(
        std::size_t epochs = 100,
        double lr = 0.001
        // std::size_t epoch_size = 20,
        // std::size_t threads = 20
    ) {
        auto optimizer = torch::optim::Adam(parameters(), lr);
        loss_fn = torch::nn::MSELoss();
        std::vector<double> loss_history;
        loss_history.reserve(epochs);
        for (std::size_t i = 0; i < epochs; i++) {
            auto graph = gen_random_graph(graph_size, connection_proba);
            Network net(graph, value_mean, value_sd, signal_life, signal_error_sd);
            auto features = get_features(net);
            auto targets = torch::tile(
                torch::from_blob(net.values),
                {graph_size, 1}
            );
            optimizer.zero_grad();
            auto loss = loss_fn(forward(features), targets);
            loss_history.push_back(loss.item<double>());
            loss.backward();
            optimizer.step();
            if ((i + 1) % epochs == 0 || i + 1 == epochs) {
                std::cout << "Epoch " << i + 1 << " of " << epochs << ": Loss = " << loss.item<double>() << '\n';
            }
        }
        return loss_history;
    }

    std::size_t graph_size;
    std::size_t max_signals;
    std::size_t signal_life;
    std::size_t signal_error_sd;
    std::size_t n_other_features;
    std::size_t n_features;
    double connection_proba;
    double value_mean;
    double value_sd;

    torch::nn::Linear first;
    std::vector<torch::nn::Linear> hidden;
    torch::nn::Linear last;

};


int main() {
    // Matrix<bool> graph(
    //     {
    //         false, true, false,
    //         true, false, true,
    //         false, true, false
    //     },
    //     3
    // );
    auto graph = gen_random_graph(3, 0.5);
    graph.print();

    Network net(graph, 0.0, 1.0, 4, 0.1);
    auto distances = net.get_distances();
    distances.print();

    net.print_signals();

    MLModel model(3);
    model.train();
}
