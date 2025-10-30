#include "DenseBlock.h"

void DenseBlock::initialize() {
    if (is_initalized) return;

    int current_channel = initial_channels;
    for (int i = 0; i < layer_count; i++) {
        for (auto& layer : layers) {
            layer->initialize(current_channel);
        }
        current_channel += growth_rate;
    }
}

Tensor<float, 4> DenseBlock::forward(const Tensor<float, 4> &input) {
    Tensor<float, 4> output = input;
    Tensor<float, 4> original_input = input;
    for (int i = 0; i < layer_count; i++) {
        original_input = output;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        output = original_input.concatenate(output, 1);
    }

    return output;
}