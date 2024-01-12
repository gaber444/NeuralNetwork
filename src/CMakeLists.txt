
add_executable(train main.cpp)
target_link_libraries(train PRIVATE classes nlohmann_json::nlohmann_json)
