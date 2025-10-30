#include "DenseLayerComposite.h"
#include "BatchNorm.h"
#include "Activation.h"
#include "ConvLayer.h"

DenseLayerComposite::DenseLayerComposite(int growth_rate, int in_channels) {
    layers.push_back(std::make_unique<BatchNorm>(in_channels));
    layers.push_back(std::make_unique<ActivationFunction>(Activation::relu));
    layers.push_back(std::make_unique<ConvLayer>(in_channels, 4 * growth_rate, 1, 1, 0));
    layers.push_back(std::make_unique<BatchNorm>(4 * growth_rate));
    layers.push_back(std::make_unique<ActivationFunction>(Activation::relu));
    layers.push_back(std::make_unique<ConvLayer>(4 * growth_rate, growth_rate, 3, 1, 1));
}

Tensor<float, 4> DenseLayerComposite::forward(const Tensor<float, 4> &input) {
    Tensor<float, 4> output = input;

    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}