#include <string>
#include <utility>
#include "ml_helper.h"

#define _USE_MATH_DEFINES
#include <cmath>

const double SQRT2PI = 2 / (M_2_SQRTPI * M_SQRT1_2);


void xavier_init(torch::nn::Module& module) {
	torch::NoGradGuard noGrad;
	if (auto* linear = module.as<torch::nn::Linear>()) {
		torch::nn::init::xavier_normal_(linear->weight);
		torch::nn::init::constant_(linear->bias, 0.01);
	}
}

std::tuple<torch::Tensor, torch::Tensor> sample_normal(
    const torch::Tensor& params
) {
    assert(params.dim() == 2 && params.size(0) == 2);
    auto mu = params[0];
    auto sigma = torch::exp(params[1]);
    auto normal_vals = torch::randn(params.size(1)) * sigma + mu;
    auto log_proba = torch::sum(-0.5 * torch::pow((normal_vals - mu) / sigma, 2) - torch::log(sigma * SQRT2PI));
    return {normal_vals, log_proba};
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
    // 1 for current time
    // total is n_goods * 4 + 1
    first = register_module(
        "first",
        torch::nn::Linear(n_goods * 4 + 1, size)
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
    for (auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    x = torch::relu(last->forward(x));
    // first row is mean; second row is log std dev
    return x.view({-1, 2, n_goods});
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
    // 1 for current time
    // total is n_goods * 5 + 1
    first = register_module(
        "first",
        torch::nn::Linear(n_goods * 5 + 1, size)
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
    for (auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    // return a single value, which is proba of acceptance
    return torch::sigmoid(last->forward(x));
}


ValueNet::ValueNet(
    int n_goods,
    int size,
    int depth
) : n_goods(n_goods) {
    // number of input params is:
    // n_goods for my util params
    // n_goods for my current goods
    // n_goods for total endowment
    // 1 for current time
    // total is n_goods * 3 + 1
    first = register_module(
        "first",
        torch::nn::Linear(n_goods * 3 + 1, size)
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

torch::Tensor ValueNet::forward(torch::Tensor x) {
    x = torch::relu(first->forward(x));
    for (auto& h : hidden) {
        x = x + torch::relu(h->forward(x));
    }
    // return a single value, which is value of current state
    return last->forward(x);
}


MLHelper::MLHelper(
    int n_goods,
    int proposing_size,
    int proposing_depth,
    int accepting_size,
    int accepting_depth,
    int value_size,
    int value_depth
) : n_goods(n_goods),
    proposingNet(
        std::make_shared<ProposingNet>(n_goods, proposing_size, proposing_depth)
    ),
    acceptingNet(
        std::make_shared<AcceptingNet>(n_goods, accepting_size, accepting_depth)
    ),
    valueNet(
        std::make_shared<ValueNet>(n_goods, value_size, value_depth)
    )
{}

MLHelper::MLHelper(
    std::shared_ptr<ProposingNet> proposingNet,
    std::shared_ptr<AcceptingNet> acceptingNet,
    std::shared_ptr<ValueNet> valueNet
) : n_goods(proposingNet->n_goods),
    proposingNet(proposingNet),
    acceptingNet(acceptingNet),
    valueNet(valueNet)
{
    assert(n_goods == acceptingNet->n_goods == valueNet->n_goods);
}


std::tuple<bool, torch::Tensor> MLHelper::accept(
    const Person& person,
    const Person& proposer,
    const torch::Tensor& offer,
    const torch::Tensor& total_endowment,
    int time
) {
    // assemble features
    auto features = torch::concat(
        {
            person.get_my_util_params(),
            person.get_goods(),
            proposer.get_goods(),
            total_endowment,
            offer,
            torch::tensor({time})
        }
    );
    // plug into AcceptingNet
    auto acceptance_proba = acceptingNet->forward(features).view({1});
    bool accept = (torch::rand(1) < acceptance_proba).item<bool>();
    auto log_proba = torch::log(
        (accept) ? acceptance_proba : 1 - acceptance_proba
    );
    return {accept, log_proba};
}


std::tuple<torch::Tensor, torch::Tensor> MLHelper::make_offer(
    const Person& proposer,
    const Person& other,
    const torch::Tensor& total_endowment,
    int time
) {
    auto features = torch::concat(
        {
            proposer.get_my_util_params(),
            proposer.get_goods(),
            other.get_goods(),
            total_endowment,
            torch::tensor({time})
        }
    );
    auto offerParams = proposingNet->forward(features).view({2, n_goods});
    return sample_normal(offerParams);
}


torch::Tensor MLHelper::get_value(
        const Person& person,
        const torch::Tensor& total_endowment,
        int time
) {
    auto features = torch::concat(
        {
            person.get_my_util_params(),
            person.get_goods(),
            total_endowment,
            torch::tensor({time})
        }
    );
    return valueNet->forward(features);
}


int MLHelper::get_n_goods() const {
    return n_goods;
}

std::vector<torch::optim::OptimizerParamGroup> MLHelper::get_params() const {
    return {proposingNet->parameters(), acceptingNet->parameters()};
}


void train(
    const torch::Tensor& util_params,
    std::shared_ptr<MLHelper> helper,
    int n_persons,
    double goods_mean,
    double goods_sd,
    int epochs,
    int steps_per_epoch,
    double lr
) {
    // util_params should be 1d tensor of length n_goods
    assert(util_params.dim() == 1);
    int n_goods = helper->get_n_goods();
    assert(util_params.size(0) == n_goods);

    UtilFunc utilFunc(util_params);
    // endowments are normal distributed
    auto goods = goods_mean + torch::randn({n_persons, n_goods}) * goods_sd;
    // everyone has same util func and helper, but different endowments
    std::vector<Person> persons;
    persons.reserve(n_persons);
    for (int i = 0; i < n_persons; i++) {
        persons.push_back(
            Person(
                goods[i],
                utilFunc,
                helper
            )
        );
    }

    ExchangeEconomy economy(persons);

    auto optim = torch::optim::Adam(helper->get_params(), lr);

    for (int i = 0; i < epochs; i++) {
        optim.zero_grad();

        auto log_probas = torch::empty(steps_per_epoch, torch::requires_grad(true));
        auto value_guesses = torch::empty(steps_per_epoch, torch::requires_grad(true));
        for (int t = 0; t < steps_per_epoch; t++) {
            auto [log_proba, value_guess] = economy.time_step();
        }

        // calculate actual value at end of trading
        double value = 0.0;
        for (const auto& person : economy.get_persons()) {
            value += person.get_consumption_util();
        }
        // translate to advantage and then to loss score
        // no reward until end of trading makes this calculation easy
        std::cout << "Vguess " << value_guesses << '\n';
        auto advantages = value_guesses - value;
        std::cout << "Adv " << advantages << '\n';
        auto loss = torch::sum(torch::pow(advantages, 2)) + torch::sum(log_probas * advantages);
        std::cout << "Loss " << loss << '\n';

        loss.backward();
        optim.step();
    }
    
}
