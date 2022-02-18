#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "networks.h"
#include "ml.h"


const int N_EVAL = 10000;
const int NET_SIZE = 10;

double get_ideal_guess(
    const std::vector<Signal>& signal_vec,
    const Matrix<int>& distances,
    double prior_mean,
    double prior_sd,
    double error_sd
) {
    double mean = prior_mean;
    double prior_precision = 1 / (prior_sd * prior_sd);
    double error_precision = 1 / (error_sd * error_sd);
    for (const Signal& signal : signal_vec) {
        int distance = 1 + distances.get(signal.sender, signal.source);
        double signal_precision = distance * error_precision;
        mean = (prior_precision * mean + signal_precision * signal.value) / (prior_precision + signal_precision);
        prior_precision = prior_precision + signal_precision;
    }
    return mean;
}

std::vector<double> get_ideal_guesses(
    const NetMember& member,
    const Matrix<int>& distances,
    double prior_mean,
    double prior_sd,
    double error_sd
) {
    std::vector<double> guesses;
    const auto& signals = member.signals();
    for (int i = 0; i < signals.size(); i++) {
        double ideal_guess;
        if (i == member.get_my_idx()) {
            ideal_guess = signals[i][0].value;
        }
        else {
            ideal_guess = get_ideal_guess(
                signals[i],
                distances,prior_mean,
                prior_sd,
                error_sd
            );
        }
        guesses.push_back(ideal_guess);
    }
    return guesses;
}


int main() {
    MLModel model(NET_SIZE, 10, 4, 100);
    model.connection_proba = 0.25;
    model.signal_life = 10;
    model.signal_error_sd = 0.2;

    model.train(2000, 0.0001, 100); 

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

    // record guesses
    torch::Tensor guesses;
    {
        torch::NoGradGuard noGrad;
        guesses = model.forward(
            torch::concat(features)
        ).index({
            torch::indexing::Slice(0, NET_SIZE*N_EVAL)
        });
    }

    // record ideal guesses according to uncorrelated bayesian
    auto ideal_guesses = torch::empty({NET_SIZE*N_EVAL, NET_SIZE});
    for (int i = 0; i < N_EVAL; i++) {
        const auto& members = networks[i].net_members();
        const Matrix<int> distances = networks[i].get_distances();
        for (int j = 0; j < NET_SIZE; j++) {
            ideal_guesses[i*NET_SIZE + j] = torch::tensor(
                get_ideal_guesses(
                    members[j],
                    distances,
                    model.value_mean,
                    model.value_sd,
                    model.signal_error_sd
                )
            );
        }
    }


    write_to_csv(
        torch::concat(
            {all_data, guesses, ideal_guesses},
            1
        ),
        "uncorrelated_bayesian_data.csv"
    );

    return 0;
}