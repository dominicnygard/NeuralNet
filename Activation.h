#ifndef ACTIVATION_H
#define ACTIVATION_H
#include <functional>
#include "Layers.h"

class ActivationFunction : public Layer {
    private:
        std::function<Tensor<double, 4>(const Tensor<double, 4>&)> activation_func;
        std::function<Tensor<double, 4>(const Tensor<double, 4>&, const Tensor<double, 4>&)> grad_func;
    public:
        explicit ActivationFunction(int input_chan, int output_chan, std::function<Tensor<double, 4>(const Tensor<double, 4>&)> activation_func,
                                std::function<Tensor<double, 4>(const Tensor<double, 4>&, const Tensor<double, 4>&)> grad_func = nullptr);

        Tensor<double, 4> forward(const Tensor<double, 4> &input) override;

        void backward(const MatrixXd& grad) {};
        void update(double learning_rate) {};
};

#endif