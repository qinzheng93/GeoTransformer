#include <cstring>
#include "grid_subsampling.h"
#include "grid_subsampling_cpu.h"

std::vector<at::Tensor> grid_subsampling(
  at::Tensor points,
  at::Tensor lengths,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_IS_FLOAT(points);
  CHECK_IS_LONG(lengths);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;

  std::vector<int64_t> vec_lengths = std::vector<int64_t>(
    lengths.data_ptr<int64_t>(),
    lengths.data_ptr<int64_t>() + batch_size
  );
  std::vector<int64_t> vec_s_lengths;

  grid_subsampling_cpu(
    vec_points,
    vec_s_points,
    vec_lengths,
    vec_s_lengths,
    voxel_size
  );

  std::size_t total_s_points = vec_s_points.size();
  at::Tensor s_points = torch::zeros(
    {static_cast<int64_t>(total_s_points), 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {static_cast<int64_t>(batch_size)},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );

  std::memcpy(
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_lengths.data_ptr<int64_t>(),
    vec_s_lengths.data(),
    sizeof(int64_t) * batch_size
  );

  return {s_points, s_lengths};
}
