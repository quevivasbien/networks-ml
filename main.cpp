#include "ml.h"

int main() {
    MLModel model(10, 4, 4, 100);
    // model.threads = 1;
    // model.connection_proba = 0.2;
    model.train(1000, 0.001, 64);
    model.eval(200, "data.csv");
}
