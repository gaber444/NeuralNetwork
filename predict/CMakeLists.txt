add_executable(predict predict.cpp)
target_link_libraries(predict PRIVATE classes nlohmann_json::nlohmann_json)
