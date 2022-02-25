#include <string>
#include <utility>
#include "ml_helper.h"

#define _USE_MATH_DEFINES
#include <cmath>


void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

std::pair<torch::Tensor, torch::Tensor> sample_normal(
    const torch::Tensor& params
) {
    assert(params.dim() == 2 && params.size(0) == 2);
    auto mu = params[0];
    auto sigma = torch::exp(params[1]);
    auto normal_vals = torch::randn(params.size(1)) * sigma + mu;
    auto log_proba = -0.5 * torch::pow((normal_vals - mu) / sigma, 2) - torch::log(sigma * SQRT2PI);
    return std::make_pair(normal_vals, log_proba);
}


ProposingNet::ProposingNet(
    int n_goods,
    int size,
    int depth
) : n_goods(n_goods) {
    // number of input params is:
    // n_goods for my util params
    // n_goods for my current goods
    // n_goods for other's current goods
    // n_goods for total endowment
    // total is n_goods * 4
    first = register_module(
        "first",
        torch::nn::Linear(n_goods * 4, size)
    );
    for (int i = 0; i < depth; i++) {
        hidden.push_back(
            register_module(
                "hidden" + std::to_string(i),
                torch::nn::Linear(size, size)
            )
        );
    }
    last = register_module(
        "last",
        torch::nn::Linear(size, 2 * n_goods)
    );

    this->apply(xavier_init);
}

torch::Tensor ProposingNet::forward(torch::Tensor x) {
    x = torch::relu(first->forward(x));
    for (const auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    x = torch::relu(last->forward(x));
    // first row is mean, second row is log std dev
    return x.view({"...", 2, n_goods});
}


AcceptingNet::AcceptingNet(
    int n_goods,
    int size,
    int depth
) : n_goods(n_goods) {
    // number of input params is:
    // n_goods for my util params
    // n_goods for my current goods
    // n_goods for other's current goods
    // n_goods for total endowment
    // n_goods for offer
    // total is n_goods * 5
    first = register_module(
        "first",
        torch::nn::Linear(n_goods * 5, size)
    );
    for (int i = 0; i < depth; i++) {
        hidden.push_back(
            register_module(
                "hidden" + std::to_string(i),
                torch::nn::Linear(size, size)
            )
        );
    }
    last = register_module(
        "last",
        torch::nn::Linear(size, 1)
    );

    this->apply(xavier_init);
}

torch::Tensor AcceptingNet::forward(torch::Tensor x) {
    x = torch::relu(first->forward(x));
    for (const auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    // return a single value, which is proba of acceptance
    x = torch::sigmoid(last->forward(x));
}


MLHelper::MLHelper(
    int n_goods,
    int proposing_size,
    int proposing_depth,
    int accepting_size,
    int accepting_depth
) : n_goods(n_goods),
    proposingNet(n_goods, proposing_size, proposing_depth),
    acceptingNet(n_goods, accepting_size, accepting_depth)
{}

MLHelper::MLHelper(
    ProposingNet proposingNet,
    AcceptingNet acceptingNet
) : n_goods(proposingNet.n_goods),
    proposingNet(proposingNet),
    acceptingNet(acceptingNet)
{
    assert(n_goods == acceptingNet.n_goods);
}


bool MLHelper::accept(
    const Person& person,
    const Person& proposer,
    const torch::Tensor& offer,
    const torch::Tensor& total_endowment
) const {
    // assemble features
    auto features = torch::stack(
        {
            person.get_my_util_params(),
            person.get_goods(),
            proposer.get_goods(),
            total_endowment,
            offer
        }
    );
    // plug into AcceptingNet
    auto acceptance_proba = acceptingNet.forward(
        features.view({1, 5*n_goods})
    ).flatten();
    bool accept = (torch::rand() < acceptance_proba).item<bool>());
    // record log proba
    log_proba = log_proba + (
        torch::log((accept) ? acceptance_proba : (1 - acceptance_proba))
    );
    return accept;
}


torch::Tensor MLHelper::make_offer(
    const Person& proposer,
    const Person& other,
    const torch::Tensor& total_endowment
) const {
    auto features = torch::stack(
        {
            proposer.get_my_util_params(),
            proposer.get_goods(),
            other.get_goods(),
            total_endowment
        }
    );
    auto out_params = proposingNet.forward(
        features.view({1, 4*n_goods})
    );
    auto offer_proba_pair = sample_normal(out_params);
    log_proba = log_proba + offer_proba_pair.second;
    return offer_proba_pair.first;
}
