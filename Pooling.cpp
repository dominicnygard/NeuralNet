#include "Pooling.h"

PoolingLayer::PoolingLayer(PoolingType type, int pool_size, int stride, Mode mode) {
    this->type = type;
    this->pool_size = pool_size;
    this->stride = stride;
    this->mode = mode;
}

Tensor<float, 4> PoolingLayer::forward(const Tensor<float, 4> &input) {
    int batch_size = input.dimension(0);
    int channels = input.dimension(1);
    int height = input.dimension(2);
    int width = input.dimension(3);

    if (mode == GLOBAL) {
        Tensor<float, 4> output(batch_size, channels, 1, 1);
        output.setZero();

        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                Tensor<float, 2> feature_map = input.chip(b, 0).chip(c, 0);
                if (type == MAX) {
                    Tensor<float, 0> max_val_scl = feature_map.maximum();
                    float max_val = max_val_scl(0);
                    output(b, c, 0, 0) = max_val;
                } else {
                    Tensor<float, 0> mean_val_scl = feature_map.mean();
                    float mean_val = mean_val_scl(0);
                    output(b, c, 0, 0) = mean_val;
                }
                    
            }
        }
        return output;
    } else {
        int output_height = (height - pool_size) / stride + 1;
        int output_width = (width - pool_size) / stride + 1;

        Tensor<float, 4> output(batch_size, channels, output_height, output_width);
        output.setZero();

        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++ ) {
                Tensor<float, 2> feature_map = input.chip(b, 0).chip(c, 0);

                for (int i = 0; i < output_height; i++) {
                    for (int j = 0; j < output_width; j++) {
                        int start_row = i * stride;
                        int start_col = j * stride;

                        int end_row = std::min(start_row + pool_size, height);
                        int end_col = std::min(start_col + pool_size, width);

                        auto slice = feature_map.slice(
                            Eigen::array<int, 2>({start_row, start_col}),
                            Eigen::array<int, 2>({end_row - start_row, end_col - start_col})
                        );

                        if (type == MAX) {
                            Tensor<float, 0> max_val_scl = slice.maximum();
                            float max_val = max_val_scl(0);
                            output(b, c, i, j) = max_val;
                        } else {
                            Tensor<float, 0> mean_val_scl = slice.mean();
                            float mean_val = mean_val_scl(0);
                            output(b, c, i, j) = mean_val;
                        }

                    }
                }
            }
        }
        return output;
    }
}