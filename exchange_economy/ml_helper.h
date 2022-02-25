#pragma once

#include <torch/torch.h>
#include <vector>
#include "exchange.h"

struct ProposingNet : torch::nn::Module {
    ProposingNet(int n_goods, int size, int depth);

    torch::Tensor forward(torch::Tensor x);

    int n_goods;
    torch::nn::Linear first = nullptr;
    std::vector<torch::nn::Linear> hidden;
    torch::nn::Linear last = nullptr;
};


struct AcceptingNet : torch::nn::Module {
    AcceptingNet(int n_goods, int size, int depth);

    torch::Tensor forward(torch::Tensor x);

    int n_goods;
    torch::nn::Linear first = nullptr;
    std::vector<torch::nn::Linear> hidden;
    torch::nn::Linear last = nullptr;
};


class MLHelper : public DecisionHelper {
public:
    MLHelper(
        int n_goods,
        int proposing_size,
        int proposing_depth,
        int accepting_size,
        int accepting_depth
    );

    MLHelper(ProposingNet proposingNet, AcceptingNet acceptingNet);

    virtual bool accept(
        const Person& person,
        const Person& proposer,
        const torch::Tensor& offer,
        const torch::Tensor& total_endowment
    ) const override;

    virtual torch::Tensor make_offer(
        const Person& proposer,
        const Person& other,
        const torch::Tensor& total_endowment
    ) const override;

private:
    int n_goods;
    ProposingNet proposingNet;
    AcceptingNet acceptingNet;
    torch::Tensor log_proba = torch::tensor(0.0);
};
