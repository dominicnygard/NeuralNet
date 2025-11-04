#ifndef LINEARLAYER_H
#define LINEARLAYER_H
#include "Layers.h"

class LinearLayer : public Layer {
private:
    Tensor<float, 2> weights;
    Tensor<float, 1> bias;
    Tensor<float, 2> dW;  // Gradient w.r.t. weights
    Tensor<float, 1> db;  // Gradient w.r.t. bias
public:
    LinearLayer(int in_features, int out_features);
    Tensor<float, 4> forward(const Tensor<float, 4>& input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY);
};

#endif