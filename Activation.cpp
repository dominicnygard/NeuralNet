#include "Activation.h"
#include <iostream>

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

Eigen::Tensor<float, 4> ActivationFunction::backward(const Eigen::Tensor<float, 4> &dY)
{
    if (!grad_func) {
        return dY;
    }

    output = grad_func(dY, input);
    return output;
}

namespace Activation {
    Tensor<float, 4> relu(const Tensor<float, 4> &input) {
        return input.cwiseMax(0.0f);
    }

    Tensor<float, 4> relu_grad(const Tensor<float, 4> &dY, const Tensor<float, 4> &x) {
        // Avoid using Tensor::select with a scalar else-value (which can
        // instantiate TensorSelectOp with a non-tensor ElseDerived and
        // trigger deep Eigen template errors). Use an element-wise mask
        // multiplication instead: dY * (x > 0).cast<float>().
        return dY * ( (x > 0.0f).cast<float>() );
    }

    Tensor<float, 4> softmax(const Tensor<float, 4> &input) {
        const int B = input.dimension(0);
        const int C = input.dimension(1);
        const int H = input.dimension(2);
        const int W = input.dimension(3);

        //std::cout << input << "\n";

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