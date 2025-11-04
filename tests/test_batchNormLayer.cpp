#include <catch2/catch_all.hpp>
#include "BatchNorm.h"
#include <cmath>

// Helper function to check if two floats are approximately equal
inline bool approx_equal_bn(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// BatchNorm Forward Tests
// =============================================================================

TEST_CASE("BatchNorm forward - basic normalization", "[batchnorm][forward]") {
    BatchNorm bn(3);
    
    Eigen::Tensor<float, 4> input(2, 3, 2, 2);
    
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;
    
    input(0, 1, 0, 0) = 5.0f;
    input(0, 1, 0, 1) = 6.0f;
    input(0, 1, 1, 0) = 7.0f;
    input(0, 1, 1, 1) = 8.0f;
    
    input(0, 2, 0, 0) = 9.0f;
    input(0, 2, 0, 1) = 10.0f;
    input(0, 2, 1, 0) = 11.0f;
    input(0, 2, 1, 1) = 12.0f;
    
    input(1, 0, 0, 0) = 2.0f;
    input(1, 0, 0, 1) = 3.0f;
    input(1, 0, 1, 0) = 4.0f;
    input(1, 0, 1, 1) = 5.0f;
    
    input(1, 1, 0, 0) = 6.0f;
    input(1, 1, 0, 1) = 7.0f;
    input(1, 1, 1, 0) = 8.0f;
    input(1, 1, 1, 1) = 9.0f;
    
    input(1, 2, 0, 0) = 10.0f;
    input(1, 2, 0, 1) = 11.0f;
    input(1, 2, 1, 0) = 12.0f;
    input(1, 2, 1, 1) = 13.0f;
    
    auto output = bn.forward(input);
    
    float expected_0_0_0_0 = -1.63298762f;
    float expected_0_0_0_1 = -0.81649375f;
    float expected_0_0_1_0 = 0.00000012f;
    float expected_0_0_1_1 = 0.81649399f;
    
    float expected_0_1_0_0 = -1.63298762f;
    float expected_0_1_0_1 = -0.81649375f;
    float expected_0_1_1_0 = 0.00000012f;
    float expected_0_1_1_1 = 0.81649399f;
    
    float expected_0_2_0_0 = -1.63298810f;
    float expected_0_2_0_1 = -0.81649423f;
    float expected_0_2_1_0 = -0.00000036f;
    float expected_0_2_1_1 = 0.81649351f;
    
    float expected_1_0_0_0 = -0.81649375f;
    float expected_1_0_0_1 = 0.00000012f;
    float expected_1_0_1_0 = 0.81649399f;
    float expected_1_0_1_1 = 1.63298786f;
    
    float expected_1_1_0_0 = -0.81649375f;
    float expected_1_1_0_1 = 0.00000012f;
    float expected_1_1_1_0 = 0.81649399f;
    float expected_1_1_1_1 = 1.63298786f;
    
    float expected_1_2_0_0 = -0.81649423f;
    float expected_1_2_0_1 = -0.00000036f;
    float expected_1_2_1_0 = 0.81649351f;
    float expected_1_2_1_1 = 1.63298738f;
    
    REQUIRE(approx_equal_bn(output(0, 0, 0, 0), expected_0_0_0_0));
    REQUIRE(approx_equal_bn(output(0, 0, 0, 1), expected_0_0_0_1));
    REQUIRE(approx_equal_bn(output(0, 0, 1, 0), expected_0_0_1_0));
    REQUIRE(approx_equal_bn(output(0, 0, 1, 1), expected_0_0_1_1));
    
    REQUIRE(approx_equal_bn(output(0, 1, 0, 0), expected_0_1_0_0));
    REQUIRE(approx_equal_bn(output(0, 1, 0, 1), expected_0_1_0_1));
    REQUIRE(approx_equal_bn(output(0, 1, 1, 0), expected_0_1_1_0));
    REQUIRE(approx_equal_bn(output(0, 1, 1, 1), expected_0_1_1_1));
    
    REQUIRE(approx_equal_bn(output(0, 2, 0, 0), expected_0_2_0_0));
    REQUIRE(approx_equal_bn(output(0, 2, 0, 1), expected_0_2_0_1));
    REQUIRE(approx_equal_bn(output(0, 2, 1, 0), expected_0_2_1_0));
    REQUIRE(approx_equal_bn(output(0, 2, 1, 1), expected_0_2_1_1));
    
    REQUIRE(approx_equal_bn(output(1, 0, 0, 0), expected_1_0_0_0));
    REQUIRE(approx_equal_bn(output(1, 0, 0, 1), expected_1_0_0_1));
    REQUIRE(approx_equal_bn(output(1, 0, 1, 0), expected_1_0_1_0));
    REQUIRE(approx_equal_bn(output(1, 0, 1, 1), expected_1_0_1_1));
    
    REQUIRE(approx_equal_bn(output(1, 1, 0, 0), expected_1_1_0_0));
    REQUIRE(approx_equal_bn(output(1, 1, 0, 1), expected_1_1_0_1));
    REQUIRE(approx_equal_bn(output(1, 1, 1, 0), expected_1_1_1_0));
    REQUIRE(approx_equal_bn(output(1, 1, 1, 1), expected_1_1_1_1));
    
    REQUIRE(approx_equal_bn(output(1, 2, 0, 0), expected_1_2_0_0));
    REQUIRE(approx_equal_bn(output(1, 2, 0, 1), expected_1_2_0_1));
    REQUIRE(approx_equal_bn(output(1, 2, 1, 0), expected_1_2_1_0));
    REQUIRE(approx_equal_bn(output(1, 2, 1, 1), expected_1_2_1_1));
}

TEST_CASE("BatchNorm forward - 1x1 spatial dimensions", "[batchnorm][forward][edge]") {
    BatchNorm bn(4);
    
    Eigen::Tensor<float, 4> input(2, 4, 1, 1);
    
    input(0, 0, 0, 0) = 1.5f;
    input(0, 1, 0, 0) = 2.5f;
    input(0, 2, 0, 0) = 3.5f;
    input(0, 3, 0, 0) = 4.5f;
    
    input(1, 0, 0, 0) = 2.0f;
    input(1, 1, 0, 0) = 3.0f;
    input(1, 2, 0, 0) = 4.0f;
    input(1, 3, 0, 0) = 5.0f;
    
    auto output = bn.forward(input);
    
    float expected_0_0 = -0.99992013f;
    float expected_0_1 = -0.99991965f;
    float expected_0_2 = -0.99992013f;
    float expected_0_3 = -0.99991965f;
    
    float expected_1_0 = 0.99991989f;
    float expected_1_1 = 0.99992037f;
    float expected_1_2 = 0.99991989f;
    float expected_1_3 = 0.99992037f;
    
    REQUIRE(approx_equal_bn(output(0, 0, 0, 0), expected_0_0));
    REQUIRE(approx_equal_bn(output(0, 1, 0, 0), expected_0_1));
    REQUIRE(approx_equal_bn(output(0, 2, 0, 0), expected_0_2));
    REQUIRE(approx_equal_bn(output(0, 3, 0, 0), expected_0_3));
    
    REQUIRE(approx_equal_bn(output(1, 0, 0, 0), expected_1_0));
    REQUIRE(approx_equal_bn(output(1, 1, 0, 0), expected_1_1));
    REQUIRE(approx_equal_bn(output(1, 2, 0, 0), expected_1_2));
    REQUIRE(approx_equal_bn(output(1, 3, 0, 0), expected_1_3));
}

// =============================================================================
// BatchNorm Backward Tests - Testing for vanishing/exploding gradients
// =============================================================================

TEST_CASE("BatchNorm backward - gradient flow", "[batchnorm][backward][gradient]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(2, 2, 2, 2);
    input.setRandom();
    input = input * 2.0f;
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(2, 2, 2, 2);
    grad_output.setRandom();
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                }
            }
        }
    }
    
    float sum = 0.0f;
    float max_abs = 0.0f;
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    sum += std::abs(grad_input(n, c, h, w));
                    max_abs = std::max(max_abs, std::abs(grad_input(n, c, h, w)));
                }
            }
        }
    }
    
    REQUIRE(sum > 0.0f);
    REQUIRE(max_abs < 100.0f);  
}

TEST_CASE("BatchNorm backward - zero variance handling", "[batchnorm][backward][edge]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(2, 2, 2, 2);
    for (int n = 0; n < 2; ++n) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                input(n, 0, h, w) = 5.0f;
                input(n, 1, h, w) = static_cast<float>(n * 4 + h * 2 + w);
            }
        }
    }
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(2, 2, 2, 2);
    grad_output.setConstant(1.0f);
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                    REQUIRE(std::isfinite(output(n, c, h, w)));
                }
            }
        }
    }
}

TEST_CASE("BatchNorm backward - extreme input values", "[batchnorm][backward][edge]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(2, 2, 2, 2);
    input.setRandom();
    input = input * 100.0f;
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(2, 2, 2, 2);
    grad_output.setConstant(1.0f);
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                    REQUIRE(std::isfinite(output(n, c, h, w)));
                }
            }
        }
    }
    
    for (int c = 0; c < 2; ++c) {
        float channel_sum = 0.0f;
        int count = 0;
        for (int n = 0; n < 2; ++n) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    channel_sum += output(n, c, h, w);
                    count++;
                }
            }
        }
        float mean = channel_sum / count;
        REQUIRE(std::abs(mean) < 1e-4f);
    }
}

TEST_CASE("BatchNorm backward - 1x1 spatial gradient flow", "[batchnorm][backward][edge]") {
    BatchNorm bn(3);
    
    Eigen::Tensor<float, 4> input(4, 3, 1, 1);
    input.setRandom();
    input = input * 5.0f;
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(4, 3, 1, 1);
    grad_output.setConstant(1.0f);
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 4; ++n) {
        for (int c = 0; c < 3; ++c) {
            REQUIRE(std::isfinite(grad_input(n, c, 0, 0)));
        }
    }
    
    float total = 0.0f;
    for (int n = 0; n < 4; ++n) {
        for (int c = 0; c < 3; ++c) {
            total += std::abs(grad_input(n, c, 0, 0));
        }
    }
    REQUIRE(total > 0.0f);
}

TEST_CASE("BatchNorm backward - small batch size", "[batchnorm][backward][edge]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(1, 2, 2, 2);
    input.setRandom();
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(1, 2, 2, 2);
    grad_output.setConstant(1.0f);
    
    auto grad_input = bn.backward(grad_output);
    
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                REQUIRE(std::isfinite(grad_input(0, c, h, w)));
                REQUIRE(std::isfinite(output(0, c, h, w)));
            }
        }
    }
}

TEST_CASE("BatchNorm backward - asymmetric gradients", "[batchnorm][backward]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(2, 2, 3, 3);
    input.setRandom();
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(2, 2, 3, 3);
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    grad_output(n, c, h, w) = 1.0f / (1.0f + h + w);
                }
            }
        }
    }
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                }
            }
        }
    }
    
    float first = grad_input(0, 0, 0, 0);
    bool has_variation = false;
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    if (std::abs(grad_input(n, c, h, w) - first) > 1e-5f) {
                        has_variation = true;
                    }
                }
            }
        }
    }
    REQUIRE(has_variation);
}

TEST_CASE("BatchNorm backward - multiple forward/backward cycles", "[batchnorm][backward]") {
    BatchNorm bn(2);
    
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Tensor<float, 4> input(2, 2, 2, 2);
        input.setRandom();
        input = input * (static_cast<float>(iter) + 1.0f);
        
        auto output = bn.forward(input);
        
        Eigen::Tensor<float, 4> grad_output(2, 2, 2, 2);
        grad_output.setConstant(1.0f);
        
        auto grad_input = bn.backward(grad_output);
        
        for (int n = 0; n < 2; ++n) {
            for (int c = 0; c < 2; ++c) {
                for (int h = 0; h < 2; ++h) {
                    for (int w = 0; w < 2; ++w) {
                        REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                        REQUIRE(std::isfinite(output(n, c, h, w)));
                    }
                }
            }
        }
    }
}

TEST_CASE("BatchNorm backward - gradient magnitude scaling", "[batchnorm][backward]") {
    BatchNorm bn(2);
    
    Eigen::Tensor<float, 4> input(2, 2, 2, 2);
    input.setRandom();
    
    auto output = bn.forward(input);
    
    Eigen::Tensor<float, 4> grad_output(2, 2, 2, 2);
    grad_output.setRandom();
    grad_output = grad_output * 1000.0f;
    
    auto grad_input = bn.backward(grad_output);
    
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    REQUIRE(std::isfinite(grad_input(n, c, h, w)));
                }
            }
        }
    }
    
    float max_grad = 0.0f;
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    max_grad = std::max(max_grad, std::abs(grad_input(n, c, h, w)));
                }
            }
        }
    }
    
    REQUIRE(max_grad > 1.0f);
    REQUIRE(max_grad < 1e6f);
}
