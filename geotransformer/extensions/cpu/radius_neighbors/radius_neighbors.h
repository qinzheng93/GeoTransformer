#pragma once

#include "../../common/torch_helper.h"

at::Tensor radius_neighbors(
  at::Tensor q_points,
  at::Tensor s_points,
  at::Tensor q_lengths,
  at::Tensor s_lengths,
  float radius
);
