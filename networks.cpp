#include "networks.h"

std::default_random_engine rng;

Matrix<bool> gen_random_graph(int size, double p) {
    assert(0.0 <= p <= 1.0);
    std::vector<bool> values(size * size);
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            bool value = dist(rng) <= p;
            values[i * size + j] = value;
            values[j * size + i] = value;
        }
    }
    return Matrix<bool>(values, size);
}

std::vector<double> sample_normal(double mean, double sd, int size) {
    std::vector<double> out;
    std::normal_distribution<double> dist(mean, sd);
    out.reserve(size);
    for (int i = 0; i < size; i++) {
        out.push_back(dist(rng));
    }
    return out;
}


NetMember::NetMember(
    int graph_size, int my_idx, double my_value
) : graph_size(graph_size),
    my_idx(my_idx),
    to_share({Signal(my_idx, my_idx, my_value)}),
    signals_(std::vector<std::vector<Signal>>(graph_size))
{
    signals_[my_idx].push_back(Signal(my_idx, my_idx, my_value));
}

void NetMember::receive_signal(Signal signal) {
    if (is_new_source_and_origin(signal)) {
        signals_[signal.source].push_back(signal);
        recently_received.push_back(signal);
    }
}

void NetMember::flush_recent_signals() {
    to_share = recently_received;
    recently_received = {};
}

void NetMember::print() const {
    std::cout << "Signals received by member " << my_idx << ":\n";
    for (const auto& signal_set : signals_) {
        for (const Signal& signal : signal_set) {
            std::cout << "Source: " << signal.source
                << ", Sender: " << signal.sender
                << ", Value: " << signal.value << '\n';
        }
    }
    std::cout << '\n';
}

bool NetMember::is_new_source_and_origin(const Signal& signal) const {
    if (signal.source == my_idx) {
        // This signal is about me, so I don't need it
        return false;
    }
    for (const Signal& my_signal : signals_[signal.source]) {
        if (my_signal.sender == signal.sender) {
            // I already have a signal about this source from this sender
            return false;
        }
    }
    return true;
}


Network::Network(
    Matrix<bool> graph,
    double value_mean,
    double value_sd,
    unsigned int signal_life,
    double signal_error_sd
) : graph_(graph),
    signal_life(signal_life),
    signal_error_sd(signal_error_sd),
    values_(sample_normal(value_mean, value_sd, graph.size())) 
{
    set_signals();
}

Matrix<int> Network::get_distances() const {
    int size = graph_.size();
    // organize index of neighbors & create a distances matrix based on graph
    std::vector<int> values;
    values.reserve(size * size);
    std::vector<std::vector<int>> neighbors(size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            bool graph_val = graph_.get(i, j);
            int distance = (i == j) ? 0 : ((graph_val) ? 1 : -1);
            values.push_back(distance);
            if (graph_.get(i, j)) {
                neighbors[i].push_back(j);
            }
        }
    }
    Matrix<int> distances(values, size);

    unsigned int n = 1;
    bool keep_going = true;
    while (keep_going) {
        keep_going = false;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
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

void Network::print_signals() const {
    for (const NetMember& member : net_members_) {
        member.print();
    }
}

void Network::set_signals() {
    // get things set up
    int size = graph_.size();
    std::vector<NetMember> members;
    members.reserve(size);
    for (int i = 0; i < size; i++) {
        members.push_back(NetMember(size, i, values_[i]));
    }
    std::normal_distribution<double> noise_dist(0.0, signal_error_sd);

    // now fill in signals
    // loop through time periods
    for (unsigned int t = 0; t < signal_life; t++) {
        // loop through signal senders
        for (int i = 0; i < size; i++) {
            const auto& signals_to_share = members[i].get_signals_to_share();
            // loop through signal receivers
            for (int j = 0; j < size; j++) {
                if (i == j || !graph_.get(i, j)) {
                    continue;
                }
                for (const Signal& s : signals_to_share) {
                    members[j].receive_signal(Signal(i, s.source, s.value + noise_dist(rng)));
                }
            }
            members[i].flush_recent_signals();
        }
    }

    net_members_ = members;
}