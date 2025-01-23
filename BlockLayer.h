#ifndef DENSELAYER_H
#define DENSELAYER_H
#include "DenseConstruct.h"

class DenseLayerComposite {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        std::vector<std::unique_ptr<ConstructInfo>> layer_constructors;
        int growth_rate;
        int bottleneck_val;
        bool is_initialized = false;
    public:
        DenseLayerComposite(int growth_rate, int bottleneck, std::vector<std::unique_ptr<ConstructInfo>> constructors)
            : layer_constructors(std::move(constructors)), growth_rate(growth_rate), bottleneck_val(bottleneck) {}

        void initialize(int in_channels);

        Tensor<double, 4> forward(const Tensor<double, 4> &input);
};

#endif