set(TARGET head-graft)

add_executable(${TARGET} head-graft.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)

target_link_libraries(${TARGET} PRIVATE llama)

install(TARGETS ${TARGET} RUNTIME)
