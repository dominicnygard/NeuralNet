#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "Layers.h"

class BatchNorm : public Layer {
    private:
        Tensor<float, 1> weights;
        Tensor<float, 1> biases;
        const float epsilon = 1e-6f;

    public:
        BatchNorm(int in_channels);

        Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

        void backward(const MatrixXf& grad) override {}
        void update(float learnign_rate) override {}
};

#endif