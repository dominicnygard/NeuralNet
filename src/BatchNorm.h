#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "Layers.h"

class BatchNorm : public Layer {
private:
    Tensor<float, 1> weights;
    Tensor<float, 1> bias;
    Tensor<float, 4> normalized_x;
    Tensor<float, 4> mean;
    Tensor<float, 4> var;
    Tensor<float, 1> mean1d;
    Tensor<float, 1> var1d;
    const float epsilon = 1e-5f;

public:
    BatchNorm(int in_channels);

    Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) override;
};

#endif