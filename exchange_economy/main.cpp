#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "ml_helper.h"
#include "exchange.h"


int main() {
    auto util_params = torch::tensor({0.5, 0.5});
    auto helper = std::make_shared<MLHelper>(2, 50, 4, 50, 4, 50, 2);

    Scenario scenario(
        util_params,
        helper,
        2, // n_persons
        1.0, // goods_mean
        0.1, // goods_sd
        4 // steps_per_epoch
    );

    scenario.train(
        100, // epochs
        16, // threadcount
        1e-5 // lr
    );

    return 0;
}