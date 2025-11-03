#include "BatchNorm.h"
#include <cmath>

BatchNorm::BatchNorm(int in_channels) {
    weights = Tensor<float, 1>(in_channels);
    bias = Tensor<float, 1>(in_channels);

    weights.setConstant(1.0f);
    bias.setConstant(0.0f);
}

Tensor<float, 4> BatchNorm::forward(const Tensor<float, 4> &input) {
    int B = input.dimension(0);
    int C = input.dimension(1);
    int H = input.dimension(2);
    int W = input.dimension(3);
    float spatial_size = B * H * W;

    this->input = input;

    // compute per-channel mean and variance across B,H,W in a vectorized way
    Eigen::array<int, 3> reduction_dims({0, 2, 3});
    Eigen::array<ptrdiff_t, 4> bcast({B, 1, H, W});
    Eigen::array<ptrdiff_t, 4> resize({1, C, 1, 1});

    // sum over batch, height and width -> gives Tensor<1> of length C
    Tensor<float, 1> mean_1d = (input.sum(reduction_dims) / spatial_size).eval();

    // broadcast mean to 4D and store
    Tensor<float, 4> mean_bc = mean_1d.reshape(resize).broadcast(bcast).eval();
    this->mean = mean_bc;
    this->mean1d = mean_1d;

    // variance: mean of squared deviations
    Tensor<float, 1> var_1d = ((input - mean_bc).square().sum(reduction_dims) / spatial_size).eval();
    Tensor<float, 4> var_bc = var_1d.reshape(resize).broadcast(bcast).eval();
    this->var = var_bc;
    this->var1d = var_1d;

    Tensor<float, 4> std_dev = (var_bc + epsilon).sqrt().eval();

    Tensor<float, 4> norm = ((input - mean_bc) / std_dev).eval();
    this->normalized_x = norm;

    return (norm * weights.reshape(resize).broadcast(bcast)).eval() + bias.reshape(resize).broadcast(bcast);
    

    /*
    for (int c = 0; c < C; ++c) {
        float mean = 0.0f;
        float var = 0.0f;

        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    mean += input(b, c, i, j);
        mean /= spatial_size;
        this->mean(c) = mean;

        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float diff = input(b, c, i, j) - mean;
                    var += diff * diff;
                }
        var /= spatial_size;
        this->var(c) = var;
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        

        for (int b = 0; b < B; ++b)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float norm = (input(b, c, i, j) - mean) * inv_std;
                    this->normalized_x(b, c, i, j) = norm;
                    output(b, c, i, j) = norm * weights(c) + bias(c);
                }

    }
    return output;
    */
}

Eigen::Tensor<float, 4> BatchNorm::backward(const Eigen::Tensor<float, 4> &dY)
{
    int B = input.dimension(0);
    int C = input.dimension(1);
    int H = input.dimension(2);
    int W = input.dimension(3);
    float spatial_size = B * H * W;
    
    Eigen::array<int, 3>reduction_dims({0, 2, 3});
    Eigen::array<ptrdiff_t, 4> bcast({B, 1, H, W});
    Eigen::array<ptrdiff_t, 4> resize({1, C, 1, 1});

    Tensor<float, 1> db = dY.sum(reduction_dims).eval();
    Tensor<float, 1> dW = (dY * normalized_x).sum(reduction_dims).eval();

    Tensor<float, 4> dXnorm = (dY * weights.reshape(resize).broadcast(bcast)).eval();

    // use the stored 1D mean/var to compute per-channel scalars
    Tensor<float, 1> std_dev1d = (var1d + epsilon).sqrt().eval();

    // tmp = sum_bhw( dXnorm * (input - mean) ) -> shape [C]
    Tensor<float, 1> tmp = (dXnorm * (input - mean)).sum(reduction_dims).eval();

    // dVar = tmp * (-0.5) * (var + eps)^(-3/2)
    Tensor<float, 1> inv_pow = (var1d + epsilon).pow(-1.5f).eval();
    Tensor<float, 1> dVar = (tmp * (-0.5f)).eval() * inv_pow;

    // term1 = sum_bhw(dXnorm) * (-1/std_dev)
    Tensor<float, 1> sum_dXnorm = dXnorm.sum(reduction_dims).eval();
    Tensor<float, 1> term1 = (sum_dXnorm * (-1.0f)).eval() / std_dev1d;

    Tensor<float, 1> sum_input_minus_mean = (input - mean).sum(reduction_dims).eval();
    Tensor<float, 1> dMean = (term1 + dVar * ((-2.0f / spatial_size) * sum_input_minus_mean)).eval();

    // broadcast scalars back to 4D for elementwise computations
    Tensor<float, 4> std_dev_bc = std_dev1d.reshape(resize).broadcast(bcast).eval();

    Tensor<float, 4> dX1 = (dXnorm / std_dev_bc).eval();
    Tensor<float, 4> dX2 = (dVar.reshape(resize).broadcast(bcast) * 2.0f * (input - mean) / spatial_size).eval();
    Tensor<float, 4> dX3 = (dMean.reshape(resize).broadcast(bcast) / spatial_size).eval();

    weights -= dW * Layer::learning_rate;
    bias -= db * Layer::learning_rate;

    return (dX1 + dX2 + dX3).eval();
}
