#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <functional>
#include "Layers.h"

class ActivationFunction : public Layer {
private:
    std::function<Tensor<float, 4>(const Tensor<float, 4>&)> activation_func;
    std::function<Tensor<float, 4>(const Tensor<float, 4>&, const Tensor<float, 4>&)> grad_func;
public:
    explicit ActivationFunction(std::function<Tensor<float, 4>(const Tensor<float, 4>&)> activation_func,
                            std::function<Tensor<float, 4>(const Tensor<float, 4>&, const Tensor<float, 4>&)> grad_func = nullptr);

    Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) override { return dY; }
};

namespace Activation {
    Tensor<float, 4> relu(const Tensor<float, 4> &input);
    Tensor<float, 4> relu_grad(const Tensor<float, 4> &grad, const Tensor<float, 4> &output);
    Tensor<float, 4> softmax(const Tensor<float, 4> &input);
}

#endif