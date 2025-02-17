#include "Activation.h"

ActivationFunction::ActivationFunction(int input_chan, int output_chan, std::function<Tensor<double, 4>(const Tensor<double, 4>&)> activation_func,
        std::function<Tensor<double, 4>(const Tensor<double, 4>&, const Tensor<double, 4>&)> grad_func) {
    in_channels = input_chan;
    out_channels = output_chan;
    this->activation_func = std::move(activation_func);
    this->grad_func = std::move(grad_func);
}

Tensor<double, 4> ActivationFunction::forward(const Tensor<double, 4> &input) {
    this->input = input;
    output = activation_func(input);
    return output;
}