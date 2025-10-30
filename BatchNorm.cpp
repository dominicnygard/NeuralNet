#include "BatchNorm.h"

BatchNorm::BatchNorm(int in_channels) {
    weights = Tensor<float, 1>(in_channels);
    biases = Tensor<float, 1>(in_channels);

    weights.setConstant(1.0f);
    biases.setConstant(0.0f);
}

Tensor<float, 4> BatchNorm::forward(const Tensor<float, 4> &input) {
    int B = input.dimension(0);
    int C = input.dimension(1);
    int H = input.dimension(2);
    int W = input.dimension(3);
    int spatial_size = B * H * W;

    if (H == 1 && W == 1) {
        return input;
    }

    Tensor<float, 4> output(B, C, H, W);

    for (int c = 0; c < C; ++c) {
        // Compute mean and variance for this channel
        float mean = 0.0f;
        float var = 0.0f;

        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    mean += input(b, c, i, j);
        mean /= spatial_size;

        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float diff = input(b, c, i, j) - mean;
                    var += diff * diff;
                }
        var /= spatial_size;
        float inv_std = 1.0f / std::sqrt(var + epsilon);

        // Normalize + affine transform
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float norm = (input(b, c, i, j) - mean) * inv_std;
                    output(b, c, i, j) = norm * weights(c) + biases(c);
                }

    }
    return output;
}
