#include "ml.h"

int main() {
    MLModel model(8, 4, 4, 100);
    model.threads = 1;
    // model.connection_proba = 0.2;
    model.train(100, 0.001, 20);
    model.eval(200, "data.csv");
}
