// #include <torch/torch>
#include <cstddef>
#include <memory>
#include <vector>
#include <array>
#include <thread>
#include <random>
#include <assert.h>

#include <iostream>


std::default_random_engine rng;


template <typename T>
class Matrix {
public:
    Matrix(std::vector<T> values, std::size_t size) : values(values), size_(size) {
        assert(values.size() == size * size);
    }

    std::size_t size() const {
        return size_;
    }
    T get(std::size_t i, std::size_t j) const {
        assert(i < size_ && j < size_);
        return values[i * size_ + j];
    }
    void set(std::size_t i, std::size_t j, T new_val) {
        assert(i < size_ && j < size_);
        values[i * size_ + j] = new_val;
    }
    void print() const {
        for (std::size_t i = 0; i < size_; i++) {
            for (std::size_t j = 0; j < size_; j++) {
                std::cout << (get(i, j)) ? '1' : '0';
                if (j == size_ - 1) {
                    std::cout << '\n';
                }
                else {
                    std::cout << ' ';
                }
            }
        }
    }

private:
    std::vector<T> values;
    std::size_t size_;
};

std::vector<double> sample_normal(double mean, double sd, std::size_t size) {
    std::vector<double> out;
    std::normal_distribution<double> dist(mean, sd);
    out.reserve(size);
    for (std::size_t i = 0; i < size; i++) {
        out.push_back(dist(rng));
    }
    return out;
}


struct Signal {
    Signal(
        std::size_t sender, std::size_t source, double value
    ) : sender(sender), source(source), value(value) {}
    std::size_t sender;
    std::size_t source;
    double value;
};


class NetMember {
public:
    NetMember(
        std::size_t graph_size, std::size_t my_idx
    ) : graph_size(graph_size), my_idx(my_idx) {
        signals.resize(graph_size);
    }

    void receive_signal(Signal signal) {
        if (!is_duplicate_source(signal)) {
            signals[signal.sender].push_back(signal);
        }
    }
private:
    bool is_duplicate_source(const Signal& signal) const {
        for (const Signal& my_signal : signals[signal.sender]) {
            if (my_signal.source == signal.source) {
                return true;
            }
        }
        return false;
    }

    std::size_t graph_size;
    std::size_t my_idx;

    std::vector<std::vector<Signal>> signals;
}


class Network {
public:
    Network(
        Matrix<bool> graph,
        double value_mean,
        double value_sd,
        unsigned int signal_life,
        double signal_error_sd
    ) : graph(graph),
        signal_life(signal_life),
        signal_error_sd(signal_error_sd),
        values(sample_normal(value_mean, value_sd, graph.size())) 
    {
        set_signals();
    }

    Matrix<int> get_distances() const {
        std::size_t size = graph.size();
        // organize index of neighbors & create a distances matrix based on graph
        std::vector<int> values;
        values.reserve(size * size);
        std::vector<std::vector<std::size_t>> neighbors(size);
        for (std::size_t i = 0; i < size; i++) {
            for (std::size_t j = 0; j < size; j++) {
                bool graph_val = graph.get(i, j);
                int distance = (i == j) ? 0 : ((graph_val) ? 1 : -1);
                values.push_back(distance);
                if (graph.get(i, j)) {
                    neighbors[i].push_back(j);
                }
            }
        }
        Matrix<int> distances(values, size);

        unsigned int n = 1;
        bool keep_going = true;
        while (keep_going) {
            keep_going = false;
            for (std::size_t i = 0; i < size; i++) {
                for (std::size_t j = 0; j < size; j++) {
                    if (i == j || distances.get(i, j) != n) {
                        continue;
                    }
                    keep_going = true;
                    for (auto k : neighbors[j]) {
                        if (i != k && distances.get(i, k) == -1) {
                            distances.set(i, k, n + 1);
                        }
                    }
                }
            }
            n++;
        }
        return distances;
    }

private:
    void set_signals() {
        std::vector<std::vector<Signal>> signals(graph.size());
        std::vector<std::vector<Signal>> to_send(graph.size());
        for (std::size_t i = 0; i < graph.size(); i++) {
            to_send[i].push_back(Signal(i, i, values[i]));
        }
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> unique_sources(graph.size());
        std::normal_distribution<double> noise_dist(0.0, signal_error_sd);

        for (unsigned int t = 0; t < signal_life; t++) {
             std::vector<std::vector<Signal>> received(graph.size());
             for (std::size_t i = 0; i < graph.size(); i++) {
                 for (std::size_t j = 0; j < graph.size(); j++) {
                     if (i != j && graph.get(i, j)) {
                        std::vector<Signal> warped_signals;
                        for (const auto& s : to_send[i]) {
                            auto pair = std::make_pair<std::size_t, std::size_t>()
                            if (s.sender != j && true) {
                                warped_signals.push_back(
                                    Signal(s.sender, i, s.value + noise_dist(rng))
                                );
                            }
                        }
                     }
                 }
             }
        }
    }

    Matrix<bool> graph;
    unsigned int signal_life;
    double signal_error_sd;
    std::vector<double> values;
};


int main() {
    Matrix<bool> graph(
        {
            false, true, false,
            true, false, true,
            false, true, false
        },
        3
    );
    graph.print();

    Network net(graph, 0.0, 1.0, 4, 0.1);
    auto distances = net.get_distances();
    distances.print();
}
