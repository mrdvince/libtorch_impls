cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(cifar VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME cifar)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATHS ~/libtorch)

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/fetch_cifar10.cmake)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp
                                          src/model.cpp
                                          src/cifar10.cpp
                                          src/test.cpp
                                          src/train.cpp
                                          include/trainer.h
                                          include/model.h
                                          include/cifar10.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
option(DOWNLOAD_DATASET "Download the cifar10 dataset from the interweb" ON)

if(DOWNLOAD_DATASET)
    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})
    fetch_cifar10(${DATA_DIR})

endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)