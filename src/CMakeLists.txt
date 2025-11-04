set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -flto")

# Library sources (everything except main.cpp)
set(LIB_SOURCES
    Layers.cpp
    Activation.cpp
    BatchNorm.cpp
    DenseLayerComposite.cpp
    ConvLayer.cpp
    DenseBlock.cpp
    LinearLayer.cpp
    Pooling.cpp
)

# Create a library for the neural network components
add_library(NeuralNetLib ${LIB_SOURCES})
target_include_directories(NeuralNetLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(NeuralNetwork main.cpp)
target_link_libraries(NeuralNetwork PRIVATE NeuralNetLib)
    
