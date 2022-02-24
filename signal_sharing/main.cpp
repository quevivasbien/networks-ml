#include <iostream>
#include <sstream>
#include <string>
#include "ml.h"


int parse_to_int(const std::string& s) {
    int x;
    try {
        std::size_t idx;
        x = std::stoi(s, &idx);
        if (idx < s.size()) {
            std::cerr << "Trailing characters after number: " << s << '\n';
        }
    }
    catch (std::invalid_argument const &e) {
        std::cerr << "Invalid number: " << s << '\n';
    }
    catch (std::out_of_range const &e) {
        std::cerr << "Number out of range: " << s << '\n';
    }
    return x;
}

double parse_to_float(const std::string& s) {
    double x;
    try {
        std::size_t idx;
        x = std::stod(s, &idx);
        if (idx < s.size()) {
            std::cerr << "Trailing characters after number: " << s << '\n';
        }
    }
    catch (std::invalid_argument const &e) {
        std::cerr << "Invalid number: " << s << '\n';
    }
    catch (std::out_of_range const &e) {
        std::cerr << "Number out of range: " << s << '\n';
    }
    return x;
}


int main(int argc, char *argv[]) {
    if (argc != 10+1) {
        std::cout << "Arguments are:\n"
            << "[graph size (int)] [connection proba (float in [0,1])] [signal life (int)] "
            << "[max signals (int)] [n hidden layers (int)] [hidden layer size (int)] "
            << "[training epochs (int)] [learning rate (float)] [epoch size (int)] "
            << "[eval size (int)]\n";
        return 0;
    }

    int graph_size = parse_to_int(argv[1]);
    double connection_proba = parse_to_float(argv[2]);
    int signal_life = parse_to_int(argv[3]);
    int max_signals = parse_to_int(argv[4]);
    int n_hidden = parse_to_int(argv[5]);
    int hidden_size = parse_to_int(argv[6]);
    int epochs = parse_to_int(argv[7]);
    double lr = parse_to_float(argv[8]);
    int epoch_size = parse_to_int(argv[9]);
    int eval_size = parse_to_int(argv[10]);

    MLModel model(graph_size, max_signals, n_hidden, hidden_size);
    model.connection_proba = connection_proba;
    model.signal_life = signal_life;
    
    model.train(epochs, lr, epoch_size);
    model.eval(eval_size, "data.csv");

    return 0;
}
