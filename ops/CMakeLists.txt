cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project (TFNeuralRenderOps VERSION 0.1 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)

file(GLOB sources 
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc" 
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# exec_program(python
#   ARGS "-c 'import tensorflow as tf;print(tf.sysconfig.get_include())'"
#   OUTPUT_VARIABLE TF_INC)
# exec_program(python 
#   ARGS "-c 'import tensorflow as tf;print(tf.sysconfig.get_lib())'"
#   OUTPUT_VARIABLE TF_LIB)

set(TF_INC "/home/haya/tools/miniconda3/lib/python3.6/site-packages/tensorflow/include" 
  CACHE PATH "TensorFlow include directory here")
set(TF_LIB "/home/haya/tools/miniconda3/lib/python3.6/site-packages/tensorflow"
  CACHE PATH "TensorFlow lib directory here")

set(tf_inc ${TF_INC})
set(tf_lib ${TF_LIB})
message(STATUS "tf_inc:${tf_inc}, tf_lib:${tf_lib}")

include_directories(${tf_inc} ${tf_inc}/external/nsync/public)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  list(APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
  list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif()

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11" 
  "-expt-extended-lambda" "--expt-relaxed-constexpr"
  "-Wno-deprecated-gpu-targets" "-DGOOGLE_CUDA=1")

cuda_add_library(${PROJECT_NAME} SHARED ${sources})
cuda_add_cufft_to_target(${PROJECT_NAME})
cuda_add_cublas_to_target(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${tf_lib}/libtensorflow_framework.so)
# target_compile_definitions(${PROJECT_NAME} PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
