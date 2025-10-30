#ifndef DENSEBLOCK_H
#define DENSEBLOCK_H
#include "DenseLayerComposite.h"

class DenseBlock {
    private:
        std::vector<std::unique_ptr<Layer>> compositeLayers;
        int growth_rate;
        int layer_count;
        int initial_channels;
        bool is_initalized = false;

    public:
        DenseBlock(int in_channels, int growth_rate, int layers);

        Tensor<float, 4> forward(const Tensor<float, 4> &input);
};

#endif