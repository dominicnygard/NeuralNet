#include "LinearLayer.h"

LinearLayer::LinearLayer(int in_features, int out_features) {
    weights = Tensor<float, 2>(out_features, in_features);
    bias = Tensor<float, 1>(out_features);

    weights.setRandom();
    bias.setZero();
}

Tensor<float, 2> LinearLayer::forward(const Tensor<float, 2>& input) {
    Tensor<float, 2> output(input.dimension(0), weights.dimension(0));
    for (int n = 0; n < input.dimension(0); n++) {
        for (int o = 0; o < weights.dimension(0); o++) {
            float sum = bias(o);
            for (int i = 0; i < weights.dimension(1); i++)
                sum += input(n, i) * weights(o, i);
            output(n, o) = sum;
        }
    }
    return output;
}