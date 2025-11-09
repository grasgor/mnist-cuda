cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mnist_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# point to your LibTorch directory
set(CMAKE_PREFIX_PATH "/opt/libtorch")

find_package(Torch REQUIRED)

# include headers + source files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(train_cpu_mnist
    main.cpp
    forward_functions.cpp
    backward_functions.cpp
)
target_include_directories(train_cpu_mnist PRIVATE include)

# link against LibTorch only
target_link_libraries(train_cpu_mnist "${TORCH_LIBRARIES}")

# ensure runtime can locate libtorch shared libs
set_target_properties(train_cpu_mnist PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
    INSTALL_RPATH "/opt/libtorch/lib"
)

# recommended compile flags
target_compile_options(train_cpu_mnist PRIVATE -Wall -Wextra -Wpedantic)
