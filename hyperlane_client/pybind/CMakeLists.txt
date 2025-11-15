cmake_minimum_required(VERSION 3.18)
project(hyperlane_tensor_socket)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# pybind11 module
pybind11_add_module(hyperlane_tensor_socket
  tensor_socket.cpp
)

target_compile_options(hyperlane_tensor_socket PRIVATE -O3 -march=native)

# Installation
install(TARGETS hyperlane_tensor_socket
  LIBRARY DESTINATION hyperlane_client)
