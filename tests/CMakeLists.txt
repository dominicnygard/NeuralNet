# Test executable

file(GLOB TEST_SOURCES *.cpp)

add_executable(neural_tests ${TEST_SOURCES})

target_link_libraries(neural_tests 
    PRIVATE 
    Catch2::Catch2WithMain
    NeuralNetLib
)

target_compile_options(neural_tests PRIVATE --coverage)
target_link_options(neural_tests PRIVATE --coverage)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(neural_tests)
