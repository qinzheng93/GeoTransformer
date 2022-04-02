// Modified from https://github.com/HuguesTHOMAS/KPConv-PyTorch
#include "cloud.h"

PointXYZ max_point(std::vector<PointXYZ> points) {
  PointXYZ maxP(points[0]);

  for (auto p : points) {
    if (p.x > maxP.x) {
      maxP.x = p.x;
    }
    if (p.y > maxP.y) {
      maxP.y = p.y;
    }
    if (p.z > maxP.z) {
      maxP.z = p.z;
    }
  }

  return maxP;
}

PointXYZ min_point(std::vector<PointXYZ> points) {
  PointXYZ minP(points[0]);

  for (auto p : points) {
    if (p.x < minP.x) {
      minP.x = p.x;
    }
    if (p.y < minP.y) {
      minP.y = p.y;
    }
    if (p.z < minP.z) {
      minP.z = p.z;
    }
  }

  return minP;
}