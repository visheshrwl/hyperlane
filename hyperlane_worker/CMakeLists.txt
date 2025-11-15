cmake_minimum_required(VERSION 3.18)
project(hyperlane_worker CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable verbose logging for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(avahi-client REQUIRED)
find_package(avahi-common REQUIRED)

# ORT (ONNX Runtime) - look for onnxruntime package
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
  message(WARNING "onnxruntime not found via find_package. Will attempt manual detection.")
  set(ONNXRUNTIME_INCLUDE_DIR /usr/local/include CACHE PATH "ONNX Runtime include directory")
  set(ONNXRUNTIME_LIB_DIR /usr/local/lib CACHE PATH "ONNX Runtime lib directory")
  set(ONNXRUNTIME_LIBS onnxruntime CACHE STRING "ONNX Runtime libraries")
else()
  get_target_property(ONNXRUNTIME_INCLUDE_DIR onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
  set(ONNXRUNTIME_LIBS onnxruntime::onnxruntime)
endif()

# Proto compilation
file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../proto/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

# Source files
file(GLOB_RECURSE WORKER_SOURCES "src/*.cc" "src/*.cu")

# CUDA kernel compilation (if any .cu files)
enable_language(CUDA)

# Main worker executable
add_executable(hyperlane_worker
  ${WORKER_SOURCES}
  ${PROTO_SRCS}
  ${GRPC_SRCS}
)

target_link_libraries(hyperlane_worker
  ${ONNXRUNTIME_LIBS}
  gRPC::grpc
  gRPC::grpc++
  protobuf::libprotobuf
  ${CUDA_LIBRARIES}
  avahi-client
  avahi-common
  pthread
)

# Link against ONNX Runtime if found manually
if(NOT onnxruntime_FOUND AND ONNXRUNTIME_LIB_DIR)
  target_link_directories(hyperlane_worker PRIVATE ${ONNXRUNTIME_LIB_DIR})
endif()

# Compiler flags for optimization
target_compile_options(hyperlane_worker PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-march=native -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3>
)

# CUDA architecture selection
set_target_properties(hyperlane_worker PROPERTIES
  CUDA_ARCHITECTURES "86;89"
)

# Installation
install(TARGETS hyperlane_worker DESTINATION bin)
