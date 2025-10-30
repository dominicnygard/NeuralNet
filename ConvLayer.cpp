#include "ConvLayer.h"

ConvLayer::ConvLayer(int in_channels, int num_filters, int kernel_size, int stride, int padding)
        : kernel_size(kernel_size), num_filters(num_filters), stride(stride), padding(padding) {
            out_channels = num_filters;
            weights = Tensor<float, 4>(num_filters, in_channels, kernel_size, kernel_size);
            bias = Tensor<float, 1>(num_filters);

            weights.setRandom();
            bias.setZero();
        }

Tensor<float, 4> ConvLayer::forward(const Tensor<float, 4> &input) {
    int batch_size = input.dimension(0);
    int in_channels = input.dimension(1);
    int input_height = input.dimension(2);
    int input_width = input.dimension(3);
    
    int output_size = 0;
    if (num_filters == 1) {
        output_size = input_height;
    } else {
        output_size = (input_height + 2 * padding - kernel_size) / stride + 1;
    }

    Tensor<float, 4> output(batch_size, num_filters, output_size, output_size);

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < num_filters; oc++) {
            Tensor<float, 2> accumulated_output(output_size, output_size);
            accumulated_output.setZero();
            for (int ic = 0; ic < in_channels; ic++) {
                Tensor<float, 2> image = input.chip(b, 0).chip(ic, 0);
                Tensor<float, 2> kernel = weights.chip(oc, 0).chip(ic, 0);

                Eigen::array<std::pair<int, int>, 2> padding_amount;
                padding_amount[0] = std::make_pair(1, 1);
                padding_amount[1] = std::make_pair(1, 1);

                Tensor<float, 2>padded_image = image.pad(padding_amount);

                Eigen::array<ptrdiff_t, 2> con_dims({0, 1});
                Tensor<float, 2> convolved_image = padded_image.convolve(kernel, con_dims);
                Tensor<float, 2> output_image(output_size, output_size);

                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < output_size; j++) {
                        output_image(i, j) = convolved_image(i * stride, j * stride);
                    }
                }
                accumulated_output += output_image;
            }
            accumulated_output = accumulated_output + bias(oc);
            output.chip(b, 0).chip(oc, 0) = TensorMap<Tensor<float, 2>>(accumulated_output.data(), output_size, output_size);
        }
    }
    return output;
}