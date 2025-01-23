#include "BlockLayer.h"

void DenseLayerComposite::initialize(int in_channels) {
    if (is_initialized) return;

    int layer_in = in_channels;
    int layer_out = 0;
    for (size_t i = 0; i < layer_constructors.size(); i++) {
        if (i == 2) {
            layer_out = bottleneck_val * growth_rate;
        } else if (i == 5) {
            layer_out = growth_rate;
        } else {
            layer_out = layer_in;
        }
        layers.push_back(layer_constructors[i]->construct(layer_in, layer_out));
        layer_in = layer_out;
    }
    is_initialized = true;
}

Tensor<double, 4> DenseLayerComposite::forward(const Tensor<double, 4> &input) {
    Tensor<double, 4> output = input;

    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}