#include "Activation.h"

ActivationFunction::ActivationFunction(std::function<Tensor<float, 4>(const Tensor<float, 4>&)> activation_func,
        std::function<Tensor<float, 4>(const Tensor<float, 4>&, const Tensor<float, 4>&)> grad_func) {
    this->activation_func = std::move(activation_func);
    this->grad_func = std::move(grad_func);
}

Tensor<float, 4> ActivationFunction::forward(const Tensor<float, 4> &input) {
    this->input = input;
    output = activation_func(input);
    return output;
}

namespace Activation {
Tensor<float, 4> relu(const Tensor<float, 4> &input) {
    return input.cwiseMax(0.0f);
}

Tensor<float, 4> relu_grad(const Tensor<float, 4> &grad, const Tensor<float, 4> &output) {
    (void)grad;
    return output;
}
}