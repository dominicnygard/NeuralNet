#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <functional>
#include "Layers.h"

class ActivationFunction : public Layer {
    private:
        std::function<Tensor<float, 4>(const Tensor<float, 4>&)> activation_func;
        std::function<Tensor<float, 4>(const Tensor<float, 4>&, const Tensor<float, 4>&)> grad_func;
    public:
        explicit ActivationFunction(int input_chan, int output_chan, std::function<Tensor<float, 4>(const Tensor<float, 4>&)> activation_func,
                                std::function<Tensor<float, 4>(const Tensor<float, 4>&, const Tensor<float, 4>&)> grad_func = nullptr);

        Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

        void backward(const MatrixXd& grad) {};
        void update(float learning_rate) {};
};

#endif