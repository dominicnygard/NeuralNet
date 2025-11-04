#include "LinearLayer.h"

LinearLayer::LinearLayer(int in_features, int out_features) {
    weights = Tensor<float, 2>(out_features, in_features);
    bias = Tensor<float, 1>(out_features);
    dW = Tensor<float, 2>(out_features, in_features);
    db = Tensor<float, 1>(out_features);

    float limit = std::sqrt(6.0f / (in_features + out_features));
    weights.setRandom();
    weights = weights * limit;
    bias.setZero();
    dW.setZero();
    db.setZero();
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
    this->output = output;
    return output;
}

/*
Y = W*X + b
dL/dW = dY * X^T
dL/db = dY
dX = W^T * dY
*/

Tensor<float, 4> LinearLayer::backward(const Tensor<float, 4>& dY) {
    int batch_size = input.dimension(0);
    int in_features = input.dimension(1);
    int out_features = weights.dimension(0);
    
    dW.setZero();
    db.setZero();
    
    Tensor<float, 4> dX(batch_size, in_features, 1, 1);
    dX.setZero();
    
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float dY_o = dY(b, o, 0, 0);
            
            db(o) += dY_o;
            
            for (int i = 0; i < in_features; i++) {
                dW(o, i) += dY_o * input(b, i, 0, 0);
                dX(b, i, 0, 0) += weights(o, i) * dY_o;
            }
        }
    }
    
    float batch_scale = 1.0f / batch_size;
    dW = dW * batch_scale;
    db = db * batch_scale;

    weights = weights - dW * Layer::learning_rate;
    bias = bias - db * Layer::learning_rate;
    
    return dX;
}
