# pragma once

#include <torch/torch.h>
#include <thread>
#include <mutex>
#include <tuple>
#include <fstream>

#include "networks.h"


const auto CPU_COUNT = std::thread::hardware_concurrency();

void xavier_init(torch::nn::Module& module);


template <typename T, typename U>
std::vector<T> as_type(std::vector<U> vec) {
    std::vector<T> out;
    out.reserve(vec.size());
    for (const auto& v : vec) {
        out.push_back(static_cast<T>(v));
    }
    return out;
}


void write_to_csv(
    const torch::Tensor& tensor,
    const std::string& filename
);


struct MLModel : torch::nn::Module {
    MLModel(
        int graph_size,
        int max_signals,
        int n_hidden,
        int hidden_size
    );

    torch::Tensor get_features(const Network& net);

    torch::Tensor forward(torch::Tensor x);

    void get_nets_features_targets_helper(
        std::vector<Network>* nets,
        std::vector<torch::Tensor>* features,
        std::vector<torch::Tensor>* targets
    );

    std::tuple<
        std::vector<Network>, torch::Tensor, torch::Tensor
    > get_nets_features_targets(int n, int threads);

    std::vector<double> train(
        int epochs,
        double lr,
        int epoch_size
    );

    torch::Tensor eval_single(
        const Network& network,
        const torch::Tensor& features,
        const torch::Tensor& targets
    );

    void eval(
        int n,
        std::string out_file
    );

    int graph_size;
    int max_signals;
    int signal_life = 5;
    int signal_error_sd = 0.1;
    int n_other_features;
    int n_features;
    double connection_proba = 0.5;
    double value_mean = 0.0;
    double value_sd = 1.0;
    int threads = CPU_COUNT;

    torch::nn::Linear first = nullptr;
    std::vector<torch::nn::Linear> hidden;
    torch::nn::Linear last = nullptr;

    std::mutex myMutex;
};
