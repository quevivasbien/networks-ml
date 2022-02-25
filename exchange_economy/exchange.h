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
    virtual bool accept(
        const Person& person,
        const Person& proposer,
        const torch::Tensor& offer,
        const torch::Tensor& total_endowment
    ) const = 0;
    virtual torch::Tensor& make_offer(
        const Person& proposer,
        const Person& other,
        const torch::Tensor& total_endowment
    ) const = 0;
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
    void time_step();
private:
    std::vector<Person> persons;
    int n_persons;
    int n_goods;
    torch::Tensor total_endowment;
};
