#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
rm -r build *.so
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
rm -r build *.so
python3 setup.py build_ext --inplace
cd ..
