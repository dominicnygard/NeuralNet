#include "ConvLayer.h"
#include <iostream>

ConvLayer::ConvLayer(int in_channels, int num_filters, int kernel_size, int stride, int padding)
        : kernel_size(kernel_size), num_filters(num_filters), stride(stride), padding(padding) {
            out_channels = num_filters;
            weights = Tensor<float, 4>(num_filters, in_channels, kernel_size, kernel_size);
            bias = Tensor<float, 1>(num_filters);
            dW = Tensor<float, 4>(num_filters, in_channels, kernel_size, kernel_size);
            db = Tensor<float, 1>(num_filters);

            int fan_in = in_channels * kernel_size * kernel_size;
            float bound = 1.0f / std::sqrt(fan_in);
            weights.setRandom();
            weights = weights * bound;
            bias.setZero();
        }

Tensor<float, 4> ConvLayer::forward(const Tensor<float, 4> &input) {
    int batch_size = input.dimension(0);
    int in_channels = input.dimension(1);
    int input_height = input.dimension(2);
    int input_width = input.dimension(3);
    this->input = input;
    
    int output_size = (input_height + 2 * padding - kernel_size) / stride + 1;

    Tensor<float, 4> output(batch_size, num_filters, output_size, output_size);

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < num_filters; oc++) {
            Tensor<float, 2> accumulated_output(output_size, output_size);
            accumulated_output.setZero();
            for (int ic = 0; ic < in_channels; ic++) {
                Tensor<float, 2> image = input.chip(b, 0).chip(ic, 0);
                Tensor<float, 2> kernel = weights.chip(oc, 0).chip(ic, 0);

                Eigen::array<std::pair<int, int>, 2> padding_amount;
                padding_amount[0] = std::make_pair(padding, padding);
                padding_amount[1] = std::make_pair(padding, padding);

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
            output.chip(b, 0).chip(oc, 0) = accumulated_output;
            //output.chip(b, 0).chip(oc, 0) = TensorMap<Tensor<float, 2>>(accumulated_output.data(), output_size, output_size);
        }
    }
    return output;
}

/*
dW = X convolve dY
dX = dY convolve W^rot180
db = sum(dY)
*/


Eigen::Tensor<float, 4> ConvLayer::backward(const Eigen::Tensor<float, 4> &dY)
{
    int batch_size = input.dimension(0);
    int in_channels = input.dimension(1);
    int input_height = input.dimension(2);
    int input_width = input.dimension(3);

    int output_size = (input_height + 2 * padding - kernel_size) / stride + 1;

    Tensor<float, 4> dX(batch_size, in_channels, input_height, input_width);

    dX.setZero();
    dW.setZero();
    db.setZero();

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < num_filters; oc++) {
            Tensor<float,0> sum_tensor = dY.chip(b, 0).chip(oc, 0).sum().eval();
            db(oc) += sum_tensor(0);
            for (int ic = 0; ic < in_channels; ic++) {
                Tensor<float, 2> kernel = weights.chip(oc, 0).chip(ic, 0); 
                Tensor<float, 2> image = input.chip(b, 0).chip(ic, 0);
                Tensor<float, 2> dY_kernel = dY.chip(b, 0).chip(oc, 0);

                array<bool, 2> reverse_dims({true, true});
                Tensor<float, 2> flipped_kernel = kernel.reverse(reverse_dims);

                Eigen::array<std::pair<int, int>, 2> padding_amount;
                padding_amount[0] = std::make_pair(padding, padding);
                padding_amount[1] = std::make_pair(padding, padding);

                Tensor<float, 2>padded_image = image.pad(padding_amount);


                Eigen::array<ptrdiff_t, 2> con_dims({0, 1});
                Tensor<float, 2> convolved_image = padded_image.convolve(dY_kernel, con_dims);
                Tensor<float, 2> output_image(kernel_size, kernel_size);

                for (int i = 0; i < kernel_size; i++) {
                    for (int j = 0; j < kernel_size; j++) {
                        output_image(i, j) = convolved_image(i * stride, j * stride);
                    }
                }

                Tensor<float, 2> sparse_dY((dY_kernel.dimension(0)-1)*stride+1, (dY_kernel.dimension(1)-1)*stride+1);
                sparse_dY.setZero();
                for (int i = 0; i < dY_kernel.dimension(0); i++) {
                    for (int j = 0; j < dY_kernel.dimension(1); j++) {
                        sparse_dY(i * stride, j * stride) = dY_kernel(i, j);
                    }
                }

                int dY_padding = (kernel_size - 1);
                padding_amount[0] = std::make_pair(dY_padding, dY_padding);
                padding_amount[1] = std::make_pair(dY_padding, dY_padding);
                
                Tensor<float, 2> padded_dY = sparse_dY.pad(padding_amount);

                Tensor<float, 2> convolved_dX = padded_dY.convolve(flipped_kernel, con_dims);
                Tensor<float, 2> output_dX(input_height, input_width);
                
                if (convolved_dX.dimension(0)-input_height>0) {
                    int overlap = convolved_dX.dimension(0)-input_height;
                    Eigen::array<Eigen::Index, 2> offsets;
                    if (overlap%2==1) {
                        overlap == 1 ? offsets = {0, 1} : offsets = {overlap / 2, overlap / 2 + 1};
                    } else {
                        offsets = {overlap / 2, overlap / 2};
                    }
                    Eigen::array<Eigen::Index, 2> extents = {input_height, input_width};

                    output_dX = convolved_dX.slice(offsets, extents);
                } else {
                    output_dX = convolved_dX;
                }
            

                dX.chip(b, 0).chip(ic, 0) += output_dX;
                dW.chip(oc, 0).chip(ic, 0) += output_image;
            }
        }
    }

    float batch_scale = 1.0f / batch_size;
    dW = dW * batch_scale;
    db = db * batch_scale;

    //std::cout << "Conv weights sum before=" << weights.sum() << std::endl;
    weights -= dW * Layer::learning_rate;
    bias -= db * Layer::learning_rate;
    //std::cout << "Conv weights sum after=" << weights.sum() << std::endl;

    return dX;
}
