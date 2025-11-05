#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "Layers.h"

class ConvLayer : public Layer {
private:
    Tensor<float, 1> bias;
    Tensor<float, 4> weights;
    Tensor<float, 4> dW;
    Tensor<float, 1> db;
    int kernel_size = 0;
    int num_filters = 0;
    int stride = 1;
    int padding = 1;
    int out_channels = 0;
public:
    ConvLayer(int in_channels, int num_filters, int kernel_size, int stride = 1, int padding = 1);

    Tensor<float, 4> forward(const Tensor<float, 4>& input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) override;

    void setWeights(const Tensor<float, 4>& w) {weights = w;}
    void setBias(const Tensor<float, 1>& b) {bias = b;}

    const Tensor<float, 4> getWeights() {return weights;}
    const Tensor<float, 1> getBias() {return bias;}
};

#endif