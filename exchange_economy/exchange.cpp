#include <cmath>
#include "exchange.h"

// void vector_add(
//     torch::Tensor& a,
//     const torch::Tensor& b
// ) {
//     // adds b to a, (a in place)
//     assert(a.size() == b.size());
//     for (int i = 0; i < a.size(); i++) {
//         a[i] += b[i];
//     }
// }

// void vector_subtract(
//     torch::Tensor& a,
//     const torch::Tensor& b
// ) {
//     // subtracts b from a, (a in place)
//     assert(a.size() == b.size());
//     for (int i = 0; i < a.size(); i++) {
//         a[i] -= b[i];
//     }
// }


UtilFunc::UtilFunc(torch::Tensor params) : params(params), n_goods(params.size(0)) {
    assert(params.dim() == 1);
}

double UtilFunc::eval(const torch::Tensor& goods) const {
    assert(goods.dim() == 1 && goods.size(0) == n_goods);
    return torch::prod(goods.pow(params)).item<double>();
}

int UtilFunc::get_n_goods() const {
    return n_goods;
}

const torch::Tensor& UtilFunc::get_params() const {
    return params;
}


Person::Person(
    torch::Tensor endowment, UtilFunc u, std::shared_ptr<DecisionHelper> helper
) : goods(endowment),
    u(u),
    n_goods(endowment.size(0)),
    helper(helper)
{
    assert(endowment.dim() == 1);
    assert(n_goods == u.get_n_goods());
}

int Person::get_n_goods() const {
    return n_goods;
}

const torch::Tensor& Person::get_goods() const {
    return goods;
}

const torch::Tensor& Person::get_my_util_params() const {
    return u.get_params();
}


ExchangeEconomy::ExchangeEconomy(std::vector<Person> persons) : persons(persons), n_persons(persons.size()) {
    assert(n_persons > 0);
    // make sure all persons have same number of goods
    n_goods = persons[0].n_goods;
    for (int i = 1; i < n_persons; i++) {
        assert(persons[i].n_goods == n_goods);
    }
    // figure out total endowment
    total_endowment = torch::zeros(n_goods);
    for (const Person& person : persons) {
        total_endowment += person.goods;
    }
}

void ExchangeEconomy::time_step() {
    for (int i = 0; i < n_persons; i++) {
        for (int j = 0; j < n_persons; j++) {
            if (i == j) {
                continue;
            }
            torch::Tensor proposal = persons[i].helper->make_offer(persons[i], persons[j], total_endowment);
            if (persons[j].helper->accept(persons[j], persons[i], proposal, total_endowment)) {
                persons[i].goods += proposal;
                persons[j].goods -= proposal;
            }
        }
    }
}