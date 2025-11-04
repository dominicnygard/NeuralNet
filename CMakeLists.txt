cmake_minimum_required(VERSION 3.14)
project(NeuralNetworkProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -flto")

# Coverage option (MUST be before add_subdirectory calls)
option(ENABLE_COVERAGE "Enable test coverage reporting" OFF)
if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage flags")
    add_compile_options(-O0 -g --coverage)
    add_link_options(--coverage)
endif()

# Include FetchContent for Catch2
include(FetchContent)

# Fetch Catch2 v3
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.11.0
)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        5.0.0
)
FetchContent_MakeAvailable(Eigen)

include_directories(${eigen_SOURCE_DIR})
# Add the src subdirectory
add_subdirectory(src)

# Enable testing
enable_testing()

# Add tests subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()
