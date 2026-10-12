set(TARGET mtp-trace)

add_executable(${TARGET} mtp-trace.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

target_link_libraries(${TARGET} PRIVATE llama llama-common)

install(TARGETS ${TARGET} RUNTIME)
