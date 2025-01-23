#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "Layers.h"

class ConvLayer : public Layer {
    private:
        Tensor<double, 1> bias;
        Tensor<double, 4> weights;
        MatrixXd dW;
        MatrixXd db;
        int kernel_size;
        int num_filters;
        int stride;
        int padding;
    public:
        ConvLayer(int in_channels, int num_filters, int kernel_size, int stride = 1, int padding = 1);

        Tensor<double, 4> forward(const Tensor<double, 4>& input) override;

        void backward(const MatrixXd& grad) override {}
        void update(double learning_rate) override {}
};

#endif