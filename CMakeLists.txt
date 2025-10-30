cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

set(SOURCES
    main.cpp
    Activation.cpp
    BatchNorm.cpp
    DenseLayerComposite.cpp
    ConvLayer.cpp
    DenseBlock.cpp
    Pooling.cpp
)

add_executable(NeuralNetwork ${SOURCES})

