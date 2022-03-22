#pragma once

#include <torch/torch.h>
#include <vector>
#include <cassert>
#include <memory>


class UtilFunc {
    // A simple cobb-douglas utility function

public:
    UtilFunc(torch::Tensor params);
    double eval(const torch::Tensor& goods) const;
    int get_n_goods() const;
    const torch::Tensor& get_params() const;
private:
    torch::Tensor params;
    int n_goods;
};


class Person;  // forward declaration

class DecisionHelper {
    // Person uses this to decide offers to make/accept

public:
    virtual std::tuple<bool, torch::Tensor> accept(
        const Person& person,
        const Person& proposer,
        const torch::Tensor& offer,
        const torch::Tensor& total_endowment,
        int time
    ) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor> make_offer(
        const Person& proposer,
        const Person& other,
        const torch::Tensor& total_endowment,
        int time
    ) = 0;
    virtual torch::Tensor get_value(
        const Person& person,
        const torch::Tensor& total_endowment,
        int time
    ) = 0;
};


class ExchangeEconomy;  // forward declaration

class Person {
    // has an endowment and a util function
    friend ExchangeEconomy;

public:
    Person(
        torch::Tensor endowment,
        UtilFunc u,
        std::shared_ptr<DecisionHelper> helper
    );
    int get_n_goods() const;
    const torch::Tensor& get_goods() const;
    const torch::Tensor& get_my_util_params() const;
    // calculate utility from consuming current goods
    const double get_consumption_util() const;

private:
    torch::Tensor goods;
    UtilFunc u;
    int n_goods;
    std::shared_ptr<DecisionHelper> helper;
};


class ExchangeEconomy {

public:
    ExchangeEconomy(
        std::vector<Person> persons
    );
    // takes a step in time and returns log proba of that step and guess of value of initial state
    std::tuple<torch::Tensor, torch::Tensor> time_step();

    const std::vector<Person>& get_persons() const;

private:
    std::vector<Person> persons;
    int n_persons;
    int n_goods;
    torch::Tensor total_endowment;
    int time = 0;
};
