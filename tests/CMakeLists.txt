# Test executable
add_executable(neural_tests test_main.cpp)

# Link against Catch2 and the neural network library
target_link_libraries(neural_tests 
    PRIVATE 
    Catch2::Catch2WithMain
    NeuralNetLib
)

include(Catch)
catch_discover_tests(neural_tests)
