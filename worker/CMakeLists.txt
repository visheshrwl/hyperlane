cmake_minimum_required(VERSION 3.28)
project(shardline_worker VERSION 0.1.0 LANGUAGES CXX)

option(SHARDLINE_CUDA "Report CUDA devices (requires the CUDA toolkit)" OFF)
option(SHARDLINE_GGML "Build the ggml engine that runs model layers (fetches ggml)" ON)
option(SHARDLINE_GGML_METAL "Let ggml run layers on Apple GPUs (macOS only)" ON)
option(SHARDLINE_GGML_CUDA "Let ggml run layers on NVIDIA GPUs (requires the CUDA toolkit)" OFF)
option(SHARDLINE_MDNS "Advertise the worker over mDNS through the dns_sd API" ON)
option(SHARDLINE_NATIVE "Optimise for the build machine's CPU" OFF)
option(SHARDLINE_WERROR "Treat compiler warnings as errors" OFF)
option(SHARDLINE_BUILD_TESTS "Build the unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
  enable_language(OBJCXX)
endif()

find_package(Threads REQUIRED)
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)

if(SHARDLINE_GGML)
  include(FetchContent)
  if(APPLE AND SHARDLINE_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
  else()
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
  endif()
  set(GGML_CUDA ${SHARDLINE_GGML_CUDA} CACHE BOOL "" FORCE)
  set(GGML_NATIVE ${SHARDLINE_NATIVE} CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  # ggml is built from the llama.cpp release the Phase 1 spike measured
  # (b11960, commit 8642541b). Pass -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<dir>
  # to use a local checkout instead of downloading it.
  FetchContent_Declare(llama_cpp
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11960.tar.gz
    URL_HASH SHA256=2b37ffa871595f90b94164f6c510ad9a34bd7f94da21557cc9878587e163af37
    SOURCE_SUBDIR ggml
    EXCLUDE_FROM_ALL SYSTEM)
  FetchContent_MakeAvailable(llama_cpp)
endif()

cmake_path(SET SHARDLINE_PROTO_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(SHARDLINE_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${SHARDLINE_GEN_DIR})

set(SHARDLINE_PROTOS shardline/v1/worker.proto)
set(SHARDLINE_PROTO_SRCS)
foreach(proto IN LISTS SHARDLINE_PROTOS)
  string(REGEX REPLACE "\\.proto$" "" stem ${proto})
  list(APPEND SHARDLINE_PROTO_SRCS
    ${SHARDLINE_GEN_DIR}/${stem}.pb.cc ${SHARDLINE_GEN_DIR}/${stem}.pb.h
    ${SHARDLINE_GEN_DIR}/${stem}.grpc.pb.cc ${SHARDLINE_GEN_DIR}/${stem}.grpc.pb.h)
endforeach()
list(TRANSFORM SHARDLINE_PROTOS PREPEND ${SHARDLINE_PROTO_DIR}/ OUTPUT_VARIABLE SHARDLINE_PROTO_FILES)
add_custom_command(
  OUTPUT ${SHARDLINE_PROTO_SRCS}
  COMMAND protobuf::protoc
    --proto_path=${SHARDLINE_PROTO_DIR}
    --cpp_out=${SHARDLINE_GEN_DIR}
    --grpc_out=${SHARDLINE_GEN_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${SHARDLINE_PROTO_FILES}
  DEPENDS ${SHARDLINE_PROTO_FILES} protobuf::protoc gRPC::grpc_cpp_plugin
  VERBATIM)

add_library(shardline_proto STATIC ${SHARDLINE_PROTO_SRCS})
target_include_directories(shardline_proto PUBLIC ${SHARDLINE_GEN_DIR})
target_link_libraries(shardline_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_library(shardline_core STATIC
  src/backend.cc
  src/devices.cc
  src/discovery.cc
  src/worker_node.cc
  src/worker_service.cc)
target_include_directories(shardline_core PUBLIC include)
target_link_libraries(shardline_core PUBLIC shardline_proto Threads::Threads)
target_compile_definitions(shardline_core PRIVATE
  SHARDLINE_VERSION="${PROJECT_VERSION}")
target_compile_options(shardline_core PRIVATE
  -Wall -Wextra -Wpedantic
  $<$<BOOL:${SHARDLINE_WERROR}>:-Werror>
  $<$<BOOL:${SHARDLINE_NATIVE}>:-march=native>)

if(SHARDLINE_GGML)
  target_sources(shardline_core PRIVATE src/ggml_backend.cc)
  target_link_libraries(shardline_core PUBLIC ggml)
  target_compile_definitions(shardline_core PRIVATE SHARDLINE_HAVE_GGML=1)
endif()

if(APPLE)
  target_sources(shardline_core PRIVATE src/device_metal.mm)
  set_source_files_properties(src/device_metal.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
  target_link_libraries(shardline_core PRIVATE "-framework Metal" "-framework Foundation")
  target_compile_definitions(shardline_core PRIVATE SHARDLINE_HAVE_METAL=1)
endif()

if(SHARDLINE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(shardline_core PRIVATE src/device_cuda.cc)
  target_link_libraries(shardline_core PRIVATE CUDA::cudart)
  target_compile_definitions(shardline_core PRIVATE SHARDLINE_HAVE_CUDA=1)
endif()

if(SHARDLINE_MDNS)
  target_sources(shardline_core PRIVATE src/discovery_dnssd.cc)
  target_compile_definitions(shardline_core PRIVATE SHARDLINE_HAVE_MDNS=1)
  if(NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DNSSD REQUIRED IMPORTED_TARGET avahi-compat-libdns_sd)
    target_link_libraries(shardline_core PRIVATE PkgConfig::DNSSD)
  endif()
endif()

add_executable(shardline-worker src/main.cc)
target_link_libraries(shardline-worker PRIVATE shardline_core)
target_compile_options(shardline-worker PRIVATE
  -Wall -Wextra -Wpedantic $<$<BOOL:${SHARDLINE_WERROR}>:-Werror>)

install(TARGETS shardline-worker DESTINATION bin)

if(SHARDLINE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    SYSTEM
    FIND_PACKAGE_ARGS NAMES GTest)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)
  add_executable(shardline_tests
    tests/devices_test.cc
    tests/discovery_test.cc
    tests/model_path_test.cc
    tests/worker_node_test.cc)
  target_link_libraries(shardline_tests PRIVATE shardline_core GTest::gtest_main)
  target_compile_options(shardline_tests PRIVATE
    -Wall -Wextra $<$<BOOL:${SHARDLINE_WERROR}>:-Werror>)
  if(SHARDLINE_GGML)
    target_sources(shardline_tests PRIVATE
      tests/backend_test.cc
      tests/tiny_model.cc)
    target_compile_definitions(shardline_tests PRIVATE SHARDLINE_HAVE_GGML=1)
    if(APPLE AND SHARDLINE_GGML_METAL)
      target_compile_definitions(shardline_tests PRIVATE SHARDLINE_TEST_METAL=1)
    endif()
  endif()
  if(SHARDLINE_MDNS)
    target_sources(shardline_tests PRIVATE tests/discovery_dnssd_test.cc)
    if(NOT APPLE)
      target_link_libraries(shardline_tests PRIVATE PkgConfig::DNSSD)
    endif()
  endif()
  gtest_discover_tests(shardline_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
    PROPERTIES TIMEOUT 30)
endif()
