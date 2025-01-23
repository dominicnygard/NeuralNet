#ifndef LAYERS_H
#define LAYERS_H
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Layer {
    public:
        Tensor<double, 4> output, input;
        int in_channels, out_channels;
        virtual Eigen::Tensor<double, 4> forward(const Eigen::Tensor<double, 4>& input) = 0;
        virtual void backward(const Eigen::MatrixXd& grad) = 0;
        virtual void update(double learning_rate) = 0;
        virtual ~Layer() = default;
};

#endif