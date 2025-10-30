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

    Tensor<float, 4> softmax(const Tensor<float, 4> &input) {
        const int B = input.dimension(0);
        const int C = input.dimension(1);
        const int H = input.dimension(2);
        const int W = input.dimension(3);

        Eigen::array<ptrdiff_t, 1> reduceC = {1};
        Eigen::array<ptrdiff_t, 4> reshapeDims = {B, 1, H, W};
        Eigen::array<ptrdiff_t, 4> bcast = {1, C, 1, 1};

        auto maxC3 = input.maximum(reduceC);
        auto maxC = maxC3.reshape(reshapeDims).broadcast(bcast);

        Tensor<float, 4> shifted = (input - maxC).eval();
        Tensor<float, 4> exps = shifted.exp();

        auto sumC3 = exps.sum(reduceC);
        auto sumC = sumC3.reshape(reshapeDims).broadcast(bcast);

        Tensor<float, 4> out = exps / sumC;
        return out;
    }
}