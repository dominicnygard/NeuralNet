#ifndef LINEARLAYER_H
#define LINEARLAYER_H
#include "Layers.h"

class LinearLayer : public Layer {
public:
    Tensor<float, 2> weights;
    Tensor<float, 1> bias;

    LinearLayer(int in_features, int out_features);

    Tensor<float, 2> forward(const Tensor<float, 2>& input);

};

#endif