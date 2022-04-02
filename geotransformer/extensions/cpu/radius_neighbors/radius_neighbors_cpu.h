#include <vector>
#include "../../extra/cloud/cloud.h"
#include "../../extra/nanoflann/nanoflann.hpp"

typedef nanoflann::KDTreeSingleIndexAdaptor<
  nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3
> my_kd_tree_t;

void radius_neighbors_cpu(
  std::vector<PointXYZ>& q_points,
  std::vector<PointXYZ>& s_points,
  std::vector<long>& q_lengths,
  std::vector<long>& s_lengths,
  std::vector<long>& neighbor_indices,
  float radius
);
