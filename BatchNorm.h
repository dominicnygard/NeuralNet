#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "Layers.h"

class BatchNorm : Layer {
    private:
        Tensor<double, 1> weights;
        Tensor<double, 1> biases;
        const double epsilon = 1e-6;

    public:
        BatchNorm(int in_channels, int out_channels);

        Tensor<double, 4> forward(const Tensor<double, 4> &input) override;

        void backward(const MatrixXd& grad) override {}
        void update(double learnign_rate) override {}
};

#endif