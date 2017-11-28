#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 example.cc -o example.so -shared -fPIC \
    -I$TF_INC -I$TF_INC/external/nsync/public -I/usr/local/cuda-8.0/include \
    -L$TF_LIB -ltensorflow_framework \
    -lcudart -L/usr/local/cuda-8.0/lib64/ -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
