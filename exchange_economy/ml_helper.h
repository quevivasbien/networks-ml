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


struct ValueNet : torch::nn::Module {
    ValueNet(int n_goods, int size, int depth);

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
        int accepting_depth,
        int value_size,
        int value_depth
    );

    MLHelper(
        std::shared_ptr<ProposingNet> proposingNet,
        std::shared_ptr<AcceptingNet> acceptingNet,
        std::shared_ptr<ValueNet> valueNet
    );

    virtual std::tuple<bool, torch::Tensor> accept(
        const Person& person,
        const Person& proposer,
        const torch::Tensor& offer,
        const torch::Tensor& total_endowment,
        int time
    ) override;

    virtual std::tuple<torch::Tensor, torch::Tensor> make_offer(
        const Person& proposer,
        const Person& other,
        const torch::Tensor& total_endowment,
        int time
    ) override;

    virtual torch::Tensor get_value(
        const Person& person,
        const torch::Tensor& total_endowment,
        int time
    ) override;

    int get_n_goods() const;
    std::vector<torch::optim::OptimizerParamGroup> get_params() const;

private:
    int n_goods;
    std::shared_ptr<ProposingNet> proposingNet;
    std::shared_ptr<AcceptingNet> acceptingNet;
    std::shared_ptr<ValueNet> valueNet;
    // torch::Tensor log_proba = torch::tensor(0.0);
};


void train(
    const torch::Tensor& util_params,
    std::shared_ptr<MLHelper> helper,
    int n_persons,
    double goods_mean,
    double goods_sd,
    int epochs,
    int steps_per_epoch,
    double lr
);
