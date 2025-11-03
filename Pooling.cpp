#include "Pooling.h"

PoolingLayer::PoolingLayer(PoolingType type, int pool_size, int stride, Mode mode) {
    this->type = type;
    this->pool_size = pool_size;
    this->stride = stride;
    this->mode = mode;
}

Tensor<float, 4> PoolingLayer::forward(const Tensor<float, 4> &input) {
    this->input = input;
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

/*
For global average pooling:
Y[b,c,0,0] = mean(X[b,c,:,:])
dX[b,c,h,w] = dY[b,c,0,0] / (H*W)
*/

Eigen::Tensor<float, 4> PoolingLayer::backward(const Eigen::Tensor<float, 4> &dY)
{
    int batch_size = input.dimension(0);
    int channels = input.dimension(1);
    int height = input.dimension(2);
    int width = input.dimension(3);

    Tensor<float, 4> dX(batch_size, channels, height, width);
    dX.setZero();

    float scale = 1.0f / (height * width);

    if (mode == GLOBAL) {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                float dY_val = dY(b, c, 0, 0) * scale;
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        dX(b, c, h, w) = dY_val;
                    }
                }
            }
        }
    } else {
        int output_height = (height - pool_size) / stride + 1;
        int output_width = (width - pool_size) / stride + 1;
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++ ) {
                Tensor<float, 2> feature_map = dX.chip(b, 0).chip(c, 0);

                for (int i = 0; i < output_height; i++) {
                    for (int j = 0; j < output_width; j++) {
                        int start_row = i * stride;
                        int start_col = j * stride;

                        int end_row = std::min(start_row + pool_size, height);
                        int end_col = std::min(start_col + pool_size, width);

                        Eigen::array<int, 2>offsets({start_row, start_col});
                        Eigen::array<int, 2>extents({end_row - start_row, end_col - start_col});

                        int window_h = end_row - start_row;
                        int window_w = end_col - start_col;
                        if (window_h <= 0 || window_w <= 0) continue; // nothing to do

                        if (type == MAX) {
                            // initialize to first element in window to avoid invalid indices
                            int max_r = start_row;
                            int max_s = start_col;
                            float max_val = input(b, c, max_r, max_s);

                            for (int r = 0; r < window_h; ++r) {
                                for (int s = 0; s < window_w; ++s) {
                                    int rr = start_row + r;
                                    int ss = start_col + s;
                                    float val = input(b, c, rr, ss);
                                    if (val > max_val) {
                                        max_val = val;
                                        max_r = rr;
                                        max_s = ss;
                                    }
                                }
                            }

                            dX(max_r < 0 || max_s < 0 ? 0 : b, c, max_r, max_s) += dY(b, c, i, j);
                        } else {
                            float per = dY(b,c,i,j) / float(window_h * window_w);
                            for (int r = 0; r < window_h; ++r)
                                for (int s = 0; s < window_w; ++s) {
                                    int rr = start_row + r;
                                    int ss = start_col + s;
                                    dX(b, c, rr, ss) += per;
                                }
                        }

                    }
                }
            }
        }
    }
    return dX;
}
