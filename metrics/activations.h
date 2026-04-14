// activation function header file

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

namespace activations {

inline double relu(double x) {
    return (x > 0) ? x : 0;
}

inline double leaky_relu(double x) {
    return (x > 0) ? x : 0.01 * x;  // FIXED BUG
}

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double tanh_fn(double x) {
    return std::tanh(x);
}

inline double bipolar_sigmoid(double x) {
    return (2.0 / (1.0 + std::exp(-x))) - 1;
}

inline int step_function(double x) {
    return (x > 0) ? 1 : 0;
}

}

#endif