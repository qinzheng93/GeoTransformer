#include <torch/extension.h>

#include "cpu/radius_neighbors/radius_neighbors.h"
#include "cpu/grid_subsampling/grid_subsampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CPU extensions
  m.def(
    "radius_neighbors",
    &radius_neighbors,
    "Radius neighbors (CPU)"
  );
  m.def(
    "grid_subsampling",
    &grid_subsampling,
    "Grid subsampling (CPU)"
  );
}
