#include "BatchNorm.h"

BatchNorm::BatchNorm(int in_channels, int out_channels) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    weights = Tensor<double, 1>(in_channels);
    biases = Tensor<double, 1>(in_channels);

    weights.setConstant(1.0);
    biases.setConstant(0.0);
}

Tensor<double, 4> BatchNorm::forward(const Tensor<double, 4> &input) {
    int batches = input.dimension(0);
    int channels = input.dimension(1);
    int cols = input.dimension(2);
    int rows = input.dimension(3);

    double num_elements = batches * cols * rows;

    Eigen::array<ptrdiff_t, 3> dimensions({0, 2, 3});
    Eigen::array<ptrdiff_t, 4> bcast({batches, 1, cols, rows});
    Eigen::array<ptrdiff_t, 4> resize = {1, channels, 1, 1};

    Tensor<double, 1> mean = input.mean(dimensions);

    auto broadcasted_mean = mean.reshape(resize).broadcast(bcast);

    auto center = (input - broadcasted_mean);

    auto squared_diff = center.square();

    auto variance_sum = squared_diff.sum(dimensions);

    auto variance = variance_sum/num_elements;

    auto broadcasted_var = variance.reshape(resize).broadcast(bcast);

    auto stddev = (broadcasted_var + epsilon).sqrt();

    Tensor<double, 4> normalized = center / stddev;

    return normalized * weights.reshape(resize).broadcast(bcast) + biases.reshape(resize).broadcast(bcast);
}
