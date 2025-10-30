#include "LinearLayer.h"

LinearLayer::LinearLayer(int in_features, int out_features) {
    weights = Tensor<float, 2>(out_features, in_features);
    bias = Tensor<float, 1>(out_features);

    weights.setRandom();
    bias.setZero();
}

Tensor<float, 4> LinearLayer::forward(const Tensor<float, 4>& input) {
    this->input = input;
    int batch_size = input.dimension(0);
    int in_features = input.dimension(1);
    int out_features = weights.dimension(0);

    Tensor<float, 4> output(batch_size, out_features, 1, 1);
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input(b, i, 0, 0) * weights(o, i);
            }
            output(b, o, 0, 0) = sum + bias(o);
        }
    }
    return output;
}

Tensor<float, 4> LinearLayer::backward(const MatrixXf& grad) {

}