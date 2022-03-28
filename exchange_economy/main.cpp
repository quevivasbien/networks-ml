#include <torch/torch.h>
#include <iostream>
#include <memory>
#include "ml_helper.h"
#include "exchange.h"


int main() {
    auto util_params = torch::tensor({0.5, 0.5});
    auto helper = std::make_shared<MLHelper>(2, 50, 4, 50, 4, 50, 2);

    train(
        util_params,
        helper,
        2,  // n_persons
        1.0,  // goods_mean
        0.1,  // goods_sd
        100,  // epochs
        4,  // steps_per_epoch
        4,  // threadcount
        0.001  // lr
    );

    return 0;
}