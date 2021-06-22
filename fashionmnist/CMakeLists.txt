cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(fmnist)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

option(DOWNLOAD_FASHION_MNIST "Download the fmnist dataset from the internet" ON)
if (DOWNLOAD_FASHION_MNIST)
  message(STATUS "Downloading fmnist dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_fmnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading fmnist dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(fmnist fmnist.cpp)
target_compile_features(fmnist PUBLIC cxx_range_for)
target_link_libraries(fmnist ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fmnist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fmnist>)
endif (MSVC)