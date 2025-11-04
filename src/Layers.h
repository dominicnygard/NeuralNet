#ifndef LAYERS_H
#define LAYERS_H
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Layer {
public:
    static float learning_rate;
    
    Tensor<float, 4> output, input;
    virtual Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& input) = 0;
    virtual Eigen::Tensor<float, 4> backward(const Eigen::Tensor<float, 4>& dY) = 0;

    virtual ~Layer() = default;
};



#endif