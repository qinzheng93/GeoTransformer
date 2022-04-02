#include <cstring>
#include "radius_neighbors.h"
#include "radius_neighbors_cpu.h"

at::Tensor radius_neighbors(
  at::Tensor q_points,
  at::Tensor s_points,
  at::Tensor q_lengths,
  at::Tensor s_lengths,
  float radius
) {
  CHECK_CPU(q_points);
  CHECK_CPU(s_points);
  CHECK_CPU(q_lengths);
  CHECK_CPU(s_lengths);
  CHECK_IS_FLOAT(q_points);
  CHECK_IS_FLOAT(s_points);
  CHECK_IS_LONG(q_lengths);
  CHECK_IS_LONG(s_lengths);
  CHECK_CONTIGUOUS(q_points);
  CHECK_CONTIGUOUS(s_points);
  CHECK_CONTIGUOUS(q_lengths);
  CHECK_CONTIGUOUS(s_lengths);

  std::size_t total_q_points = q_points.size(0);
  std::size_t total_s_points = s_points.size(0);
  std::size_t batch_size = q_lengths.size(0);

  std::vector<PointXYZ> vec_q_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(q_points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(q_points.data_ptr<float>()) + total_q_points
  );
  std::vector<PointXYZ> vec_s_points = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(s_points.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(s_points.data_ptr<float>()) + total_s_points
  );
  std::vector<long> vec_q_lengths = std::vector<long>(
    q_lengths.data_ptr<long>(), q_lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths = std::vector<long>(
    s_lengths.data_ptr<long>(), s_lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_neighbor_indices;

  radius_neighbors_cpu(
    vec_q_points,
    vec_s_points,
    vec_q_lengths,
    vec_s_lengths,
    vec_neighbor_indices,
    radius
  );

  std::size_t max_neighbors = vec_neighbor_indices.size() / total_q_points;

  at::Tensor neighbor_indices = torch::zeros(
    {total_q_points, max_neighbors},
    at::device(q_points.device()).dtype(at::ScalarType::Long)
  );

  std::memcpy(
    neighbor_indices.data_ptr<long>(),
    vec_neighbor_indices.data(),
    sizeof(long) * total_q_points * max_neighbors
  );

  return neighbor_indices;
}
