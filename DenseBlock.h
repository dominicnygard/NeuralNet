#ifndef DENSEBLOCK_H
#define DENSEBLOCK_H
#include "DenseLayerComposite.h"

class DenseBlock : public Layer{
private:
    std::vector<std::unique_ptr<Layer>> compositeLayers;
    std::vector<int> channel_sizes;
    int growth_rate;
    int layer_count;
    int initial_channels;
    bool is_initalized = false;

    std::vector<Tensor<float, 4>> splitChannels(const Tensor<float, 4>& concatTensor, const std::vector<int>& channelSizes);

public:
    DenseBlock(int in_channels, int growth_rate, int layers);

    Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

    Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) override;
};

#endif