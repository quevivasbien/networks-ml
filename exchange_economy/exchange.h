#pragma once

#include <vector>
#include <cassert>
#include <memory>
// #include <utility>


class UtilFunc {
    // A simple cobb-douglas utility function
public:
    UtilFunc(std::vector<double> params);
    double eval(std::vector<double> goods);
    int get_n_goods() const;
private:
    std::vector<double> params;
    int n_goods;
};


struct Offer {
    Offer(std::vector<double> give, std::vector<double> get) : give(give), get(get) {}
    const std::vector<double> give;
    const std::vector<double> get;
};


class Person;  // forward declaration

class DecisionHelper {
    // Person uses this to decide offers to make/accept
public:
    virtual bool accept(const Person& person, const Offer& offer) const = 0;
    virtual Offer make_offer(const Person& proposer, const Person& other) const = 0;
};


class ExchangeEconomy;  // forward declaration

class Person {
    // has an endowment and a util function
    friend ExchangeEconomy;
public:
    Person(
        std::vector<double> endowment,
        UtilFunc u,
        std::shared_ptr<DecisionHelper> helper
    );
    int get_n_goods() const;
    const std::vector<double>& get_goods() const;
private:
    std::vector<double> goods;
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
};
