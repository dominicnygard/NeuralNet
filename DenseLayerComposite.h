#ifndef DENSELAYER_H
#define DENSELAYER_H
#include "Layers.h"

class DenseLayerComposite : public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    DenseLayerComposite(int growth_rate, int in_channels);

    Tensor<float, 4> forward(const Tensor<float, 4> &input);
    Eigen::Tensor<float, 4> backward(const MatrixXf& grad) override {}
    void update(float learning_rate) override {}
};

#endif