#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "networks.h"
#include "ml.h"


const int N_EVAL = 10000;


int main() {
    MLModel model(2, 1, 4, 100);
    model.connection_proba = 1.0;
    model.signal_life = 1;
    model.signal_error_sd = 0.5;

    model.train(500, 0.002, 48); 

    // get eval data
    auto [networks, features, targets] = model.get_nets_features_targets(
        N_EVAL,
        model.threads
    );
    std::vector<torch::Tensor> batch_data;
    batch_data.reserve(N_EVAL);
    for (int i = 0; i < N_EVAL; i++) {
        batch_data.push_back(
            model.eval_single(
                networks[i],
                features.index(
                    {torch::indexing::Slice(i*model.graph_size, (i+1)*model.graph_size)}
                ),
                targets.index(
                    {torch::indexing::Slice(i*model.graph_size, (i+1)*model.graph_size)}
                )
            )
        );
    }
    auto all_data = torch::concat(batch_data);

    // also record true values and signals received
    auto signals = torch::empty({2*N_EVAL, 2});
    for (int i = 0; i < N_EVAL; i++) {
        const auto& members = networks[i].net_members();
        const auto& mem0signals = members[0].signals();
        const auto& mem1signals = members[1].signals();
        signals[2*i][0] = mem0signals[0][0].value;
        signals[2*i][1] = mem0signals[1][0].value;
        signals[2*i+1][0] = mem1signals[0][0].value;
        signals[2*i+1][1] = mem1signals[1][0].value;
    }

    // finally get guesses
    torch::Tensor guesses;
    {
        torch::NoGradGuard noGrad;
        guesses = model.forward(
            torch::concat(features)
        ).index({
            torch::indexing::Slice(0, 2*N_EVAL)
        });
    }

    write_to_csv(
        torch::concat(
            {all_data, signals, guesses},
            1
        ),
        "two_player_data.csv"
    );

    return 0;
}