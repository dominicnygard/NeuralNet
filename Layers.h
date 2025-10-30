#ifndef LAYERS_H
#define LAYERS_H
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Layer {
    public:
        Tensor<float, 4> output, input;
        virtual Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& input) = 0;
        virtual void backward(const Eigen::MatrixXf& grad) = 0;
        virtual void update(float learning_rate) = 0;
        virtual ~Layer() = default;
};

#endif