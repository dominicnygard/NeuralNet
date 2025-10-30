#ifndef DENSEBLOCK_H
#define DENSEBLOCK_H
#include "DenseLayerComposite.h"

class DenseBlock : public Layer{
private:
    std::vector<std::unique_ptr<Layer>> compositeLayers;
    int growth_rate;
    int layer_count;
    int initial_channels;
    bool is_initalized = false;

public:
    DenseBlock(int in_channels, int growth_rate, int layers);

    Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

    Eigen::Tensor<float, 4> backward(const MatrixXf& grad) override {}
    void update(float learning_rate) override {}
};

#endif