#include <vector>
#include "../../extra/cloud/cloud.h"
#include "../../extra/nanoflann/nanoflann.hpp"

typedef nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3
> my_kd_tree_t;

void radius_neighbors_cpu(
  std::vector<PointXYZ>& q_points,
  std::vector<PointXYZ>& s_points,
  std::vector<int64_t>& q_lengths,
  std::vector<int64_t>& s_lengths,
  std::vector<int64_t>& neighbor_indices,
  float radius
);
