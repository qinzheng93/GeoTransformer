#include "radius_neighbors_cpu.h"

void radius_neighbors_cpu(
  std::vector<PointXYZ>& q_points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& q_lengths,
  std::vector<long>& s_lengths,
  std::vector<long>& neighbor_indices,
  float radius
) {
  std::size_t i0 = 0;
  float r2 = radius * radius;

  std::size_t max_count = 0;
  std::vector<std::vector<std::pair<std::size_t, float>>> all_inds_dists(
    q_points.size()
  );

  std::size_t b = 0;
  std::size_t q_start_index = 0;
  std::size_t s_start_index = 0;

  PointCloud current_cloud;
  current_cloud.pts = std::vector<PointXYZ>(
    s_points.begin() + s_start_index,
    s_points.begin() + s_start_index + s_lengths[b]
  );

  nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10);
  my_kd_tree_t* index = new my_kd_tree_t(3, current_cloud, tree_params);;
  index->buildIndex();

  nanoflann::SearchParams search_params;
  search_params.sorted = true;

  for (auto& p0 : q_points) {
    if (i0 == q_start_index + q_lengths[b]) {
      q_start_index += q_lengths[b];
      s_start_index += s_lengths[b];
      b++;

      current_cloud.pts.clear();
      current_cloud.pts = std::vector<PointXYZ>(
        s_points.begin() + s_start_index,
        s_points.begin() + s_start_index + s_lengths[b]
      );

      delete index;
      index = new my_kd_tree_t(3, current_cloud, tree_params);
      index->buildIndex();
    }

    all_inds_dists[i0].reserve(max_count);
    float query_pt[3] = {p0.x, p0.y, p0.z};
    std::size_t nMatches = index->radiusSearch(
      query_pt, r2, all_inds_dists[i0], search_params
    );

    if (nMatches > max_count) {
      max_count = nMatches;
    }

    i0++;
  }

  delete index;

  neighbor_indices.resize(q_points.size() * max_count);
  i0 = 0;
  s_start_index = 0;
  q_start_index = 0;
  b = 0;
  for (auto& inds_dists : all_inds_dists) {
    if (i0 == q_start_index + q_lengths[b]) {
      q_start_index += q_lengths[b];
      s_start_index += s_lengths[b];
      b++;
    }

    for (std::size_t j = 0; j < max_count; j++) {
      std::size_t i = i0 * max_count + j;
      if (j < inds_dists.size()) {
        neighbor_indices[i] = inds_dists[j].first + s_start_index;
      } else {
        neighbor_indices[i] = s_points.size();
      }
    }

    i0++;
  }
}
