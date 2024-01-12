cmake_minimum_required(VERSION 3.20.0)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(json)

add_subdirectory(classes)
add_subdirectory(src)
add_subdirectory(predict)
