set(TARGET draft-logits)

add_executable(${TARGET} draft-logits.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)

target_link_libraries(${TARGET} PRIVATE llama)

install(TARGETS ${TARGET} RUNTIME)
