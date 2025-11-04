#include <catch2/catch_all.hpp>
#include "Activation.h"
#include <cmath>

bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// ReLU Tests
// =============================================================================

TEST_CASE("ReLU forward - positive values unchanged", "[activation][relu]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setConstant(5.0f);
    
    auto result = Activation::relu(input);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(result(0, c, h, w) == 5.0f);
            }
        }
    }
}

TEST_CASE("ReLU forward - negative values become zero", "[activation][relu]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setConstant(-3.5f);
    
    auto result = Activation::relu(input);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(result(0, c, h, w) == 0.0f);
            }
        }
    }
}

TEST_CASE("ReLU forward - mixed positive and negative", "[activation][relu]") {
    Eigen::Tensor<float, 4> input(1, 3, 2, 2);
    input.setConstant(-1.0f);
    input(0, 0, 0, 0) = 5.0f;
    input(0, 1, 1, 1) = 3.0f;
    input(0, 2, 0, 1) = 0.0f;
    
    auto result = Activation::relu(input);
    
    REQUIRE(result(0, 0, 0, 0) == 5.0f);
    REQUIRE(result(0, 1, 1, 1) == 3.0f);
    REQUIRE(result(0, 2, 0, 0) == 0.0f);
    REQUIRE(result(0, 2, 0, 1) == 0.0f);
    REQUIRE(result(0, 0, 1, 0) == 0.0f);
}

TEST_CASE("ReLU forward - zero input", "[activation][relu]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setZero();
    
    auto result = Activation::relu(input);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(result(0, c, h, w) == 0.0f);
            }
        }
    }
}

TEST_CASE("ReLU gradient - positive input passes gradient", "[activation][relu][gradient]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setConstant(5.0f);
    
    Eigen::Tensor<float, 4> grad(1, 2, 2, 2);
    grad.setConstant(2.0f);
    
    auto result = Activation::relu_grad(grad, input);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(result(0, c, h, w) == 2.0f);
            }
        }
    }
}

TEST_CASE("ReLU gradient - negative input blocks gradient", "[activation][relu][gradient]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setConstant(-3.0f);
    
    Eigen::Tensor<float, 4> grad(1, 2, 2, 2);
    grad.setConstant(2.0f);
    
    auto result = Activation::relu_grad(grad, input);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(result(0, c, h, w) == 0.0f);
            }
        }
    }
}

TEST_CASE("ReLU gradient - mixed input", "[activation][relu][gradient]") {
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input(0, 0, 0, 0) = 5.0f;
    input(0, 0, 0, 1) = -2.0f;
    input(0, 0, 1, 0) = 0.0f;
    input(0, 0, 1, 1) = 3.0f;
    input(0, 1, 0, 0) = -1.0f;
    input(0, 1, 0, 1) = 7.0f;
    input(0, 1, 1, 0) = -5.0f;
    input(0, 1, 1, 1) = 1.0f;
    
    Eigen::Tensor<float, 4> grad(1, 2, 2, 2);
    grad.setConstant(3.0f);
    
    auto result = Activation::relu_grad(grad, input);
    
    REQUIRE(result(0, 0, 0, 0) == 3.0f); 
    REQUIRE(result(0, 0, 0, 1) == 0.0f);  
    REQUIRE(result(0, 0, 1, 0) == 0.0f); 
    REQUIRE(result(0, 0, 1, 1) == 3.0f);  
    REQUIRE(result(0, 1, 0, 0) == 0.0f);  
    REQUIRE(result(0, 1, 0, 1) == 3.0f);  
    REQUIRE(result(0, 1, 1, 0) == 0.0f);   
    REQUIRE(result(0, 1, 1, 1) == 3.0f);   
}

// =============================================================================
// Softmax Tests
// =============================================================================

TEST_CASE("Softmax - outputs sum to 1", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 5, 1, 1);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0) = 2.0f;
    input(0, 2, 0, 0) = 3.0f;
    input(0, 3, 0, 0) = 4.0f;
    input(0, 4, 0, 0) = 5.0f;
    
    auto result = Activation::softmax(input);
    
    float sum = 0.0f;
    for (int c = 0; c < 5; ++c) {
        sum += result(0, c, 0, 0);
        REQUIRE(result(0, c, 0, 0) > 0.0f); 
        REQUIRE(result(0, c, 0, 0) < 1.0f);  
    }
    REQUIRE(approx_equal(sum, 1.0f));
}

TEST_CASE("Softmax - uniform input gives uniform output", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 4, 1, 1);
    input.setConstant(2.0f);
    
    auto result = Activation::softmax(input);
    
    float expected = 1.0f / 4.0f;  
    for (int c = 0; c < 4; ++c) {
        REQUIRE(approx_equal(result(0, c, 0, 0), expected));
    }
}

TEST_CASE("Softmax - max input has highest probability", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 5, 1, 1);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0) = 2.0f;
    input(0, 2, 0, 0) = 10.0f;  
    input(0, 3, 0, 0) = 1.5f;
    input(0, 4, 0, 0) = 2.5f;
    
    auto result = Activation::softmax(input);
    
    for (int c = 0; c < 5; ++c) {
        if (c != 2) {
            REQUIRE(result(0, 2, 0, 0) > result(0, c, 0, 0));
        }
    }
}

TEST_CASE("Softmax - numerical stability with large values", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 3, 1, 1);
    input(0, 0, 0, 0) = 1000.0f;
    input(0, 1, 0, 0) = 1001.0f;
    input(0, 2, 0, 0) = 1002.0f;
    
    auto result = Activation::softmax(input);
    
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) {
        REQUIRE(std::isfinite(result(0, c, 0, 0)));
        sum += result(0, c, 0, 0);
    }
    REQUIRE(approx_equal(sum, 1.0f));
}

TEST_CASE("Softmax - handles negative values", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 4, 1, 1);
    input(0, 0, 0, 0) = -5.0f;
    input(0, 1, 0, 0) = -2.0f;
    input(0, 2, 0, 0) = 0.0f;
    input(0, 3, 0, 0) = 3.0f;
    
    auto result = Activation::softmax(input);
    
    float sum = 0.0f;
    for (int c = 0; c < 4; ++c) {
        REQUIRE(result(0, c, 0, 0) > 0.0f);
        sum += result(0, c, 0, 0);
    }
    REQUIRE(approx_equal(sum, 1.0f));
    
    REQUIRE(result(0, 3, 0, 0) > result(0, 2, 0, 0));
    REQUIRE(result(0, 2, 0, 0) > result(0, 1, 0, 0));
    REQUIRE(result(0, 1, 0, 0) > result(0, 0, 0, 0));
}

TEST_CASE("Softmax - batch processing", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(2, 3, 1, 1);

    input(0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0) = 2.0f;
    input(0, 2, 0, 0) = 3.0f;

    input(1, 0, 0, 0) = 5.0f;
    input(1, 1, 0, 0) = 5.0f;
    input(1, 2, 0, 0) = 5.0f;
    
    auto result = Activation::softmax(input);
    
    float sum0 = result(0, 0, 0, 0) + result(0, 1, 0, 0) + result(0, 2, 0, 0);
    float sum1 = result(1, 0, 0, 0) + result(1, 1, 0, 0) + result(1, 2, 0, 0);
    
    REQUIRE(approx_equal(sum0, 1.0f));
    REQUIRE(approx_equal(sum1, 1.0f));
    
    REQUIRE(approx_equal(result(1, 0, 0, 0), 1.0f/3.0f));
    REQUIRE(approx_equal(result(1, 1, 0, 0), 1.0f/3.0f));
    REQUIRE(approx_equal(result(1, 2, 0, 0), 1.0f/3.0f));
}

TEST_CASE("Softmax - spatial dimensions", "[activation][softmax]") {
    Eigen::Tensor<float, 4> input(1, 3, 2, 2); 
    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            input(0, 0, h, w) = 1.0f;
            input(0, 1, h, w) = 2.0f;
            input(0, 2, h, w) = 3.0f;
        }
    }
    
    auto result = Activation::softmax(input);
    
    for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
            float sum = result(0, 0, h, w) + result(0, 1, h, w) + result(0, 2, h, w);
            REQUIRE(approx_equal(sum, 1.0f));
        }
    }
}

// =============================================================================
// ActivationFunction Wrapper Tests
// =============================================================================

TEST_CASE("ActivationFunction - ReLU forward and backward", "[activation][wrapper]") {
    ActivationFunction relu_layer(Activation::relu, Activation::relu_grad);
    
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input(0, 0, 0, 0) = 3.0f;
    input(0, 0, 0, 1) = -2.0f;
    input(0, 0, 1, 0) = 5.0f;
    input(0, 0, 1, 1) = -1.0f;
    input(0, 1, 0, 0) = 0.0f;
    input(0, 1, 0, 1) = 7.0f;
    input(0, 1, 1, 0) = -4.0f;
    input(0, 1, 1, 1) = 2.0f;
    
    auto output = relu_layer.forward(input);
    
    REQUIRE(output(0, 0, 0, 0) == 3.0f);
    REQUIRE(output(0, 0, 0, 1) == 0.0f);
    REQUIRE(output(0, 0, 1, 0) == 5.0f);
    REQUIRE(output(0, 0, 1, 1) == 0.0f);
    
    Eigen::Tensor<float, 4> grad(1, 2, 2, 2);
    grad.setConstant(1.0f);
    
    auto grad_out = relu_layer.backward(grad);
    
    REQUIRE(grad_out(0, 0, 0, 0) == 1.0f);  
    REQUIRE(grad_out(0, 0, 0, 1) == 0.0f);  
    REQUIRE(grad_out(0, 0, 1, 0) == 1.0f);  
    REQUIRE(grad_out(0, 0, 1, 1) == 0.0f);  
}

TEST_CASE("ActivationFunction - Softmax without gradient function", "[activation][wrapper]") {
    ActivationFunction softmax_layer(Activation::softmax);
    
    Eigen::Tensor<float, 4> input(1, 3, 1, 1);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0) = 2.0f;
    input(0, 2, 0, 0) = 3.0f;
    
    auto output = softmax_layer.forward(input);
    
    float sum = output(0, 0, 0, 0) + output(0, 1, 0, 0) + output(0, 2, 0, 0);
    REQUIRE(approx_equal(sum, 1.0f));
    
    Eigen::Tensor<float, 4> grad(1, 3, 1, 1);
    grad.setConstant(0.5f);
    
    auto grad_out = softmax_layer.backward(grad);
    
    REQUIRE(grad_out(0, 0, 0, 0) == 0.5f);
    REQUIRE(grad_out(0, 1, 0, 0) == 0.5f);
    REQUIRE(grad_out(0, 2, 0, 0) == 0.5f);
}

TEST_CASE("ActivationFunction - multiple forward passes", "[activation][wrapper]") {
    ActivationFunction relu_layer(Activation::relu, Activation::relu_grad);
    
    Eigen::Tensor<float, 4> input1(1, 2, 1, 1);
    input1(0, 0, 0, 0) = 5.0f;
    input1(0, 1, 0, 0) = -3.0f;
    
    auto output1 = relu_layer.forward(input1);
    REQUIRE(output1(0, 0, 0, 0) == 5.0f);
    REQUIRE(output1(0, 1, 0, 0) == 0.0f);
    
    Eigen::Tensor<float, 4> input2(1, 2, 1, 1);
    input2(0, 0, 0, 0) = -2.0f;
    input2(0, 1, 0, 0) = 7.0f;
    
    auto output2 = relu_layer.forward(input2);
    REQUIRE(output2(0, 0, 0, 0) == 0.0f);
    REQUIRE(output2(0, 1, 0, 0) == 7.0f);
}
