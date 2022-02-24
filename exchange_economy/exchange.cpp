#include <cmath>
#include "exchange.h"

void vector_add(
    std::vector<double>& a,
    const std::vector<double>& b
) {
    // adds b to a, (a in place)
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a[i] += b[i];
    }
}

void vector_subtract(
    std::vector<double>& a,
    const std::vector<double>& b
) {
    // subtracts b from a, (a in place)
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        a[i] -= b[i];
    }
}


UtilFunc::UtilFunc(std::vector<double> params) : params(params), n_goods(params.size()) {}

double UtilFunc::eval(std::vector<double> goods) {
    assert(goods.size() == n_goods);
    double out = 1.0;
    for (int i = 0; i < n_goods; i++) {
        out *= std::pow(goods[i], params[i]);
    }
    return out;
}

int UtilFunc::get_n_goods() const {
    return n_goods;
}


Person::Person(
    std::vector<double> endowment, UtilFunc u, std::shared_ptr<DecisionHelper> helper
) : goods(endowment),
    u(u),
    n_goods(endowment.size()),
    helper(helper)
{
    assert(n_goods == u.get_n_goods());
}

int Person::get_n_goods() const {
    return n_goods;
}

const std::vector<double>& Person::get_goods() const {
    return goods;
}


ExchangeEconomy::ExchangeEconomy(std::vector<Person> persons) : persons(persons), n_persons(persons.size()) {
    // make sure all persons have same number of goods
    if (n_persons > 0) {
        n_goods = persons[0].n_goods;
        for (int i = 1; i < n_persons; i++) {
            assert(persons[i].n_goods == n_goods);
        }
    }
}

void ExchangeEconomy::time_step() {
    for (int i = 0; i < n_persons; i++) {
        for (int j = 0; j < n_persons; j++) {
            if (i == j) {
                continue;
            }
            Offer proposal = persons[i].helper->make_offer(persons[i], persons[j]);
            if (persons[j].helper->accept(persons[j], proposal)) {
                vector_add(persons[i].goods, proposal.get);
                vector_subtract(persons[i].goods, proposal.give);
                vector_add(persons[j].goods, proposal.give);
                vector_subtract(persons[j].goods, proposal.get);
            }
        }
    }
}
