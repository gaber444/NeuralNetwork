set(all_classes
    neuron.cpp
    layer.cpp
    matrix.cpp
    neuralNetwork.cpp
    utils.cpp)

add_library(classes ${all_classes})
target_include_directories(classes PUBLIC .)
target_link_libraries(classes PUBLIC nlohmann_json::nlohmann_json)