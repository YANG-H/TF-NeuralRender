#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

NVCC=/usr/local/cuda/bin/nvcc 
CXX=g++

$NVCC -std=c++11 example.cu -o example.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
    -I$TF_INC -I$TF_INC/external/nsync/public \
    --expt-relaxed-constexpr --expt-extended-lambda
$CXX -std=c++11 example.cc example.cu.o -o example.so -shared -fPIC \
    -I$TF_INC -I$TF_INC/external/nsync/public -I/usr/local/cuda/include \
    -L$TF_LIB -ltensorflow_framework \
    -lcudart -L/usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
