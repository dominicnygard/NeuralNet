#ifndef DENSEBLOCK_H
#define DENSEBLOCK_H
#include "BlockLayer.h"

class DenseBlock {
    private:
        std::vector<std::unique_ptr<DenseLayerComposite>> layers;
        int growth_rate;
        int layer_count;
        int bottleneck_val;
        int initial_channels;
        bool is_initalized = false;

    public:
        DenseBlock(int input_channels, int growth_rate, int bottleneck, int layers) 
            : growth_rate(growth_rate), initial_channels(input_channels), bottleneck_val(bottleneck), layer_count(layers) {}

        template<typename... LayerInfos>
        void addCompositeLayer(LayerInfos... infos) {
            std::vector<std::unique_ptr<ConstructInfo>> constructors;
            (constructors.push_back(std::move(infos)), ...);
            
            for (int i = 0; i < layer_count; i++) {
                layers.push_back(std::make_unique<DenseLayerComposite>(growth_rate, bottleneck_val, std::move(constructors)));
            }
        }

        void initialize();
        Tensor<float, 4> forward(const Tensor<float, 4> &input);
};

#endif