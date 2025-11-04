#include <catch2/catch_test_macros.hpp>
#include "Activation.h"
#include "LinearLayer.h"

TEST_CASE("Activation ReLU works correctly", "[activation]") {
    Eigen::Tensor<float, 4> input(1, 3, 2, 2);
    input.setConstant(-1.0f);
    input(0, 0, 0, 0) = 5.0f;
    input(0, 1, 1, 1) = 3.0f;
    
    auto relu_result = Activation::relu(input);
    
    REQUIRE(relu_result(0, 0, 0, 0) == 5.0f);
    REQUIRE(relu_result(0, 1, 1, 1) == 3.0f);
    REQUIRE(relu_result(0, 2, 0, 0) == 0.0f);  // Negative became 0
}

TEST_CASE("Basic tensor operations", "[tensor]") {
    Eigen::Tensor<float, 4> t(2, 3, 4, 5);
    t.setConstant(1.0f);
    
    REQUIRE(t.dimension(0) == 2);
    REQUIRE(t.dimension(1) == 3);
    REQUIRE(t.dimension(2) == 4);
    REQUIRE(t.dimension(3) == 5);
    REQUIRE(t(0, 0, 0, 0) == 1.0f);
}

// Example test for LinearLayer
TEST_CASE("LinearLayer initialization", "[linear]") {
    LinearLayer layer(10, 5);
    
    // Test that layer can perform forward pass
    Eigen::Tensor<float, 4> input(1, 10, 1, 1);
    input.setConstant(1.0f);
    
    auto output = layer.forward(input);
    
    REQUIRE(output.dimension(0) == 1);
    REQUIRE(output.dimension(1) == 5);
    REQUIRE(output.dimension(2) == 1);
    REQUIRE(output.dimension(3) == 1);
}
