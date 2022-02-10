#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <iostream>


template <typename T>
class Matrix {
public:
    Matrix(std::vector<T> values, int size) : values_(values), size_(size) {
        assert(values_.size() == size * size);
    }

    int size() const {
        return size_;
    }

    T get(int i, int j) const {
        assert(i < size_ && j < size_);
        return values_[i * size_ + j];
    }

    void set(int i, int j, T new_val) {
        assert(i < size_ && j < size_);
        values_[i * size_ + j] = new_val;
    }

    const std::vector<T>& values() const {
        return values_;
    }

    void print() const {
        for (int i = 0; i < size_; i++) {
            for (int j = 0; j < size_; j++) {
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
    std::vector<T> values_;
    int size_;
};

Matrix<bool> gen_random_graph(int size, double p);

std::vector<double> sample_normal(double mean, double sd, int size);


struct Signal {
    Signal(
        int sender, int source, double value
    ) : sender(sender), source(source), value(value) {}
    int sender;
    int source;
    double value;
};


class NetMember {
public:
    NetMember(
        int graph_size, int my_idx, double my_value
    );

    void receive_signal(Signal signal);

    void flush_recent_signals();

    int get_my_idx() const {
        return my_idx;
    }

    const std::vector<Signal>& get_signals_to_share() const {
        return to_share;
    }

    const std::vector<std::vector<Signal>>& signals() const {
        return signals_;
    }

    void print() const;

private:
    bool is_new_source_and_origin(const Signal& signal) const;

    int graph_size;
    int my_idx;

    std::vector<std::vector<Signal>> signals_;
    std::vector<Signal> to_share;
    std::vector<Signal> recently_received;
};


class Network {
public:
    Network(
        Matrix<bool> graph,
        double value_mean,
        double value_sd,
        unsigned int signal_life,
        double signal_error_sd
    );

    Matrix<int> get_distances() const;

    void print_signals() const;

    const Matrix<bool>& graph() const {
        return graph_;
    }

    const std::vector<double>& values() const {
        return values_;
    }

    const std::vector<NetMember>& net_members() const {
        return net_members_;
    }

private:
    void set_signals();

    Matrix<bool> graph_;
    unsigned int signal_life;
    double signal_error_sd;
    std::vector<double> values_;
    std::vector<NetMember> net_members_;
};
