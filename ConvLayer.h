#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "Layers.h"

class ConvLayer : public Layer {
private:
    Tensor<float, 1> bias;
    Tensor<float, 4> weights;
    Tensor<float, 4> dW;
    Tensor<float, 1> db;
    int kernel_size;
    int num_filters;
    int stride;
    int padding;
    int out_channels;
public:
    ConvLayer(int in_channels, int num_filters, int kernel_size, int stride = 1, int padding = 1);

    Tensor<float, 4> forward(const Tensor<float, 4>& input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) override;
};

#endif