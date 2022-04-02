#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                                         \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x)                                                          \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x)                                                   \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                        \
  CHECK_CUDA(x);                                                              \
  CHECK_CONTIGUOUS(x)

#define CHECK_IS_INT(x)                                                       \
  do {                                                                        \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                       \
                #x " must be an int tensor");                                 \
  } while (0)

#define CHECK_IS_LONG(x)                                                      \
  do {                                                                        \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Long,                      \
                #x " must be an long tensor");                                \
  } while (0)

#define CHECK_IS_FLOAT(x)                                                     \
  do {                                                                        \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                     \
                #x " must be a float tensor");                                \
  } while (0)
