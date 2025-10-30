#include "DenseBlock.h"

DenseBlock::DenseBlock(int input_channels, int growth_rate, int layers) 
    : growth_rate(growth_rate), initial_channels(input_channels), layer_count(layers) {

    for(int i = 1; i <= layers; i++) {
        compositeLayers.push_back(std::make_unique<DenseLayerComposite>(growth_rate, input_channels + growth_rate*(i-1)));
    }
}


Tensor<float, 4> DenseBlock::forward(const Tensor<float, 4> &input) {
    Tensor<float, 4> concatenated = input;

    for (auto& layer : compositeLayers) {
        Tensor<float, 4> new_features = layer->forward(concatenated);
        Tensor<float, 4> temp = concatenated.concatenate(new_features, 1);
        concatenated = temp;
    }
    return concatenated;
}

