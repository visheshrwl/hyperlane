cmake_minimum_required(VERSION 3.25)
project(shardline_spike_stage LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.cache/llama.cpp" CACHE PATH
    "llama.cpp checkout providing ggml (pinned tag: see docs/architecture.md)")
add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

add_executable(stage_probe stage.cc stage_probe.cc)
target_link_libraries(stage_probe PRIVATE ggml)
target_compile_options(stage_probe PRIVATE -Wall -Wextra)
