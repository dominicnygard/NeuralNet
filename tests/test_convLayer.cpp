#include <catch2/catch_all.hpp>
#include "test_expected_values.h"
#include "ConvLayer.h"

inline Tensor<float, 4> createWeights(int output_channels, int input_channels, int kernel_h, int kernel_w) {
    const int total = output_channels * input_channels * kernel_h * kernel_w;
    Tensor<float, 4> weights(output_channels, input_channels, kernel_h, kernel_w);
    int idx = 0;
    for (int o = 0; o < output_channels; o++) {
        for (int c = 0; c <input_channels; c++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    float v = (idx / float(total - 1)) - 0.5f; // in [-0.5, 0.5]
                    weights(o, c, kh, kw) = v;
                    idx++;
                }
            }
        }
    }
    return weights;
}

// =============================================================================
// Convolution layer Forward Tests
// =============================================================================


TEST_CASE("Conv forward stride = 1 padding = 1 kernel = 3 batch = 2", "[convolution][forward]") {

    int batches = 2;
    int in_channels = 3;
    int out_channels = 8;
    int in_h = 10;
    int in_w = 10;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = 0; n < batches; ++n) {
        for (int h = 0; h < in_h; ++h) {
            for (int w = 0; w < in_w; ++w) {
                input(n, 0, h, w) = 5.0f;
                input(n, 1, h, w) = static_cast<float>(n * 4 + h * 2 + w);
                input(n, 2, h, w) = 1.0f;
            }
        }
    }

    Tensor<float, 1> bias(8);
    bias.setValues({0.4f, 0.2f, 0.5f, 0.5f, 0.2f, 0.9f, 0.7f, 0.5f});
    Tensor<float, 4> weights = createWeights(out_channels, in_channels, 3, 3);

    auto expectedOutput = forwardConvolution::conv3x3_stride1_pad1_batch2();

    auto convolution = ConvLayer(3, 8, 3, 1, 1);
    convolution.setBias(bias);
    convolution.setWeights(weights);

    Tensor<float, 4> output = convolution.forward(input);

    const float eps = 1e-5f;
    for (int b = 0; b < batches; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int h = 0; h < expectedOutput.dimension(2); h++) {
                for (int w  = 0; w < expectedOutput.dimension(3); w++) {
                    const float actual = output(b, oc, h, w);
                    const float expected = expectedOutput(b, oc, h, w);
                    INFO("b=" << b << ", oc=" << oc << ", h=" << h << ", w=" << w
                         << ", actual=" << actual << ", expected=" << expected
                         << ", diff=" << (actual - expected));
                    REQUIRE(actual == Catch::Approx(expected).margin(eps));
                }
            }
        }
    }

}


TEST_CASE("Conv forward stride = 2 padding = 1 kernel = 3 batch = 2", "[convolution][forward]") {
    const int batches = 2;
    const int in_channels = 3;
    const int out_channels = 4;
    const int in_h = 10, in_w = 10;
    const int kernel = 3, stride = 2, padding = 1;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = 0; n < batches; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    if (c == 0) input(n, c, h, w) = 5.0f;
                    else if (c == 1) input(n, c, h, w) = static_cast<float>(n * 4 + h * 2 + w);
                    else input(n, c, h, w) = 1.0f;
                }
            }
        }
    }

    Tensor<float, 1> bias(out_channels);
    bias.setValues({0.1f, 0.2f, 0.3f, 0.4f});
    Tensor<float, 4> weights = createWeights(out_channels, in_channels, kernel, kernel);

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setBias(bias);
    conv.setWeights(weights);

    Tensor<float, 4> output = conv.forward(input);

    const int out_h = (in_h + 2 * padding - kernel) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel) / stride + 1;
    REQUIRE(output.dimension(2) == out_h);
    REQUIRE(output.dimension(3) == out_w);

    Tensor<float, 4> expectedOutput = forwardConvolution::conv3x3_stride2_pad1_batch2();

    for (int b = 0; b < batches; ++b)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int h = 0; h < out_h; ++h)
                for (int w = 0; w < out_w; ++w)
                    REQUIRE(output(b, oc, h, w) == Catch::Approx(expectedOutput(b, oc, h, w)).margin(1e-5f));
}

TEST_CASE("Conv forward stride = 1 padding = 2 kernel = 3 batch = 2", "[convolution][forward]") {
    const int batches = 2;
    const int in_channels = 3;
    const int out_channels = 5;
    const int in_h = 10, in_w = 10;
    const int kernel = 3, stride = 1, padding = 2; // padding > 1

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = 0; n < batches; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    if (c == 0) input(n, c, h, w) = 5.0f;
                    else if (c == 1) input(n, c, h, w) = static_cast<float>(n * 3 + h + w);
                    else input(n, c, h, w) = 2.0f;
                }
            }
        }
    }

    Tensor<float, 1> bias(out_channels);
    bias.setValues({0.5f, 0.4f, 0.3f, 0.2f, 0.1f});
    Tensor<float, 4> weights = createWeights(out_channels, in_channels, kernel, kernel);

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setBias(bias);
    conv.setWeights(weights);

    Tensor<float, 4> output = conv.forward(input);
    const int out_h = (in_h + 2 * padding - kernel) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel) / stride + 1;
    REQUIRE(out_h == 12);
    REQUIRE(out_w == 12);
    REQUIRE(output.dimension(2) == out_h);
    REQUIRE(output.dimension(3) == out_w);

    Tensor<float, 4> expectedOutput = forwardConvolution::conv3x3_stride1_pad2_batch2();

    for (int b = 0; b < batches; ++b)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int h = 0; h < out_h; ++h)
                for (int w = 0; w < out_w; ++w)
                    REQUIRE(output(b, oc, h, w) == Catch::Approx(expectedOutput(b, oc, h, w)).margin(1e-5f));
}

TEST_CASE("Conv forward stride = 1 padding = 0 kernel = 1 batch = 1", "[convolution][forward]") {
    const int batches = 1;
    const int in_channels = 1;
    const int out_channels = 4;
    const int in_h = 8, in_w = 8;
    const int kernel = 1, stride = 1, padding = 0;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int h = 0; h < in_h; ++h) {
        for (int w = 0; w < in_w; ++w) {
            input(0, 0, h, w) = static_cast<float>(h * 2 + w);
        }
    }

    Tensor<float, 1> bias(out_channels);
    bias.setValues({0.1f, 0.2f, 0.3f, 0.4f});
    Tensor<float, 4> weights = createWeights(out_channels, in_channels, kernel, kernel);

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setBias(bias);
    conv.setWeights(weights);

    Tensor<float, 4> output = conv.forward(input);
    REQUIRE(output.dimension(2) == in_h);
    REQUIRE(output.dimension(3) == in_w);

    Tensor<float, 4> expectedOutput = forwardConvolution::conv1x1_stride1_pad0_batch1();
    const int out_h = (in_h + 2 * padding - kernel) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel) / stride + 1;

    for (int b = 0; b < batches; ++b)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int h = 0; h < out_h; ++h)
                for (int w = 0; w < out_w; ++w)
                    REQUIRE(output(b, oc, h, w) == Catch::Approx(expectedOutput(b, oc, h, w)).margin(1e-5f));
}

TEST_CASE("Conv forward stride = 1 padding = 0 kernel = 1 batch = 2", "[convolution][forward]") {
    const int batches = 2;
    const int in_channels = 4;
    const int out_channels = 2;
    const int in_h = 10, in_w = 10;
    const int kernel = 1, stride = 1, padding = 0;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = 0; n < batches; ++n) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    input(n, c, h, w) = static_cast<float>(n * 0.5f + c * 1.0f + h * 0.25f + w * 0.1f);
                }
            }
        }
    }

    Tensor<float, 1> bias(out_channels);
    bias.setValues({0.3f, -0.2f});
    Tensor<float, 4> weights = createWeights(out_channels, in_channels, kernel, kernel);

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setBias(bias);
    conv.setWeights(weights);

    Tensor<float, 4> output = conv.forward(input);
    REQUIRE(output.dimension(2) == in_h);
    REQUIRE(output.dimension(3) == in_w);

    Tensor<float, 4> expectedOutput = forwardConvolution::conv1x1_stride1_pad0_batch2();
    const int out_h = (in_h + 2 * padding - kernel) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel) / stride + 1;

    for (int b = 0; b < batches; ++b)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int h = 0; h < out_h; ++h)
                for (int w = 0; w < out_w; ++w)
                    REQUIRE(output(b, oc, h, w) == Catch::Approx(expectedOutput(b, oc, h, w)).margin(1e-5f));
}

TEST_CASE("Conv forward zero weights and zero bias produce zero output", "[convolution][forward]") {
    const int batches = 2, in_channels = 3, out_channels = 4;
    const int in_h = 6, in_w = 6, kernel = 3, stride = 1, padding = 1;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = 0; n < batches; ++n)
        for (int c = 0; c < in_channels; ++c)
            for (int h = 0; h < in_h; ++h)
                for (int w = 0; w < in_w; ++w)
                    input(n, c, h, w) = static_cast<float>((n+1) * (c+1) + h + w);

    Tensor<float, 4> zeroW(out_channels, in_channels, kernel, kernel);
    zeroW.setZero();
    Tensor<float, 1> zeroB(out_channels);
    zeroB.setZero();

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setWeights(zeroW);
    conv.setBias(zeroB);

    Tensor<float, 4> output = conv.forward(input);
    for (int b = 0; b < batches; ++b)
        for (int oc = 0; oc < out_channels; ++oc)
            for (int h = 0; h < output.dimension(2); ++h)
                for (int w = 0; w < output.dimension(3); ++w)
                    REQUIRE(output(b, oc, h, w) == Catch::Approx(0.0f).margin(1e-7f));
}

TEST_CASE("Conv forward zero weights, non-zero bias -> constant maps", "[convolution][forward]") {
    const int batches = 1, in_channels = 2, out_channels = 3;
    const int in_h = 5, in_w = 7, kernel = 3, stride = 1, padding = 1;

    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int c = 0; c < in_channels; ++c)
        for (int h = 0; h < in_h; ++h)
            for (int w = 0; w < in_w; ++w)
                input(0, c, h, w) = static_cast<float>(c + h + w);

    Tensor<float, 4> zeroW(out_channels, in_channels, kernel, kernel);
    zeroW.setZero();
    Tensor<float, 1> bias(out_channels);
    bias.setValues({0.25f, -1.0f, 3.5f});

    auto conv = ConvLayer(in_channels, out_channels, kernel, stride, padding);
    conv.setWeights(zeroW);
    conv.setBias(bias);

    Tensor<float, 4> output = conv.forward(input);
    for (int oc = 0; oc < out_channels; ++oc)
        for (int h = 0; h < output.dimension(2); ++h)
            for (int w = 0; w < output.dimension(3); ++w)
                REQUIRE(output(0, oc, h, w) == Catch::Approx(bias(oc)).margin(1e-7f));
}
