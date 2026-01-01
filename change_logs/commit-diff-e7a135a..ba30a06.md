# Commit Diff: e7a135a..ba30a06

- Date: 2025-12-17
- Range: `e7a135a` → `ba30a06`

## Commits
- Previous: `e7a135a` — Create LICENSE (Author: Zheng Qin; 2022-12-05)
- Current:  `ba30a06` — update to incldue type hint and update top-k parts to address sparse point cloud isse and update parts to avoid encountering detach issue yielded from tpe is in fact list (Author: JeffeeH; 2025-12-17)

## Summary
- Files changed: 15
- Insertions: 2155
- Deletions: 823

## Change Overview
- Type hints and signatures: Added `typing` annotations (`Optional`, `Union`, `Tuple`) across ops, registration, transforms, and utils for clearer APIs.
- Device handling: Replaced hardcoded `.cuda()` with device-aware tensors; added optional `out_device` to move CPU extension outputs to a target device.
- Numpy/torch inputs: Functions now accept both `torch.Tensor` and NumPy arrays and coerce to safe dtype/device/contiguity before calling C++ ops.
- KNN/top-k safety: Partition and KNN routines guard `k` against available points; maintain fixed `point_limit` output with mask/sentinel padding for sparse clouds.
- Transform helpers: Extended transformation utilities (rotation/transform helpers surfaced and annotated) and clarified error messages.
- Mask and shape consistency: Consistent boolean masks and fixed shapes for downstream modules; comments/docstrings updated accordingly.

## Effects and Impact
- Robustness: Handles sparse point clouds without index errors by safely reducing `k` and padding results; downstream code can rely on fixed shapes plus masks.
- Interop: Accepts NumPy inputs seamlessly; reduces friction at dataset/augmentation boundaries.
- Portability: Removes hard GPU assumptions; code respects current tensor device and can run on CPU or GPU more reliably.
- Clarity: Type hints improve editor tooling and reduce misuse; docstrings outline accepted types and return shapes.
- Ext-op flow: Explicit CPU staging for C++ ops, with optional move to `out_device`, makes data movement explicit and easier to profile.

## Behavioral Changes to Note
- KNN/partition outputs: Always `(M, K)` where `K == point_limit`; entries beyond available neighbors are padded (masked or sentinel). Callers should honor the returned masks.
- Devices: Tensors are created on the relevant `device` instead of calling `.cuda()` unconditionally; avoid relying on implicit CUDA placement.
- Input coercion: Non-tensor inputs are coerced to tensors with `float32/long` dtypes and contiguous CPU memory prior to C++ calls.

## Validation Checklist
- Run unit/integration paths that consume KNN/partition results and verify masking logic is used (no out-of-bounds or invalid indices).
- Validate end-to-end registration on a sparse subset to confirm no top-k errors and comparable metrics.
- If running on CPU-only, exercise ops to ensure device handling is correct; if on GPU, confirm `out_device` routing as expected.

## File Changes (additions, deletions)
- `geotransformer/modules/ops/grid_subsample.py`: +31, -14
- `geotransformer/modules/ops/pairwise_distance.py`: +6, -2
- `geotransformer/modules/ops/pointcloud_partition.py`: +60, -27
- `geotransformer/modules/ops/radius_search.py`: +51, -17
- `geotransformer/modules/ops/transformation.py`: +19, -11
- `geotransformer/modules/ops/vector_angle.py`: +4, -2
- `geotransformer/modules/registration/matching.py`: +101, -69
- `geotransformer/modules/registration/metrics.py`: +126, -88
- `geotransformer/transforms/functional.py`: +270, -125
- `geotransformer/utils/common.py`: +34, -20
- `geotransformer/utils/data.py`: +290, -160
- `geotransformer/utils/pointcloud.py`: +508, -134
- `geotransformer/utils/registration.py`: +395, -112
- `geotransformer/utils/torch.py`: +14, -9
- `geotransformer/utils/visualization.py`: +246, -33

## Module-by-Module Changes
- `modules/registration/metrics.py`:
	- Refactor to torch-native computations; remove numpy loops for anisotropic errors.
	- Add `_rotation_angle_from_relative` and `_rotation_error_angle_deg` helpers.
	- Functions now return tensors (or tuples) with reduction semantics: "mean"|"sum"|"none".
	- Effects: GPU-friendly metrics; numerically stable angle computation; cleaner APIs.

- `modules/registration/matching.py`:
	- Device-awareness (no hard `.cuda()`); added `Union[...]` return types with optional scores.
	- `get_node_correspondences` and downstream use consistent masks and device; improved readability.
	- Overlap/occlusion ratio functions now typed; broadcasting and mask math clarified.
	- Effects: Safer execution across CPU/GPU; clearer shapes and optional outputs.

- `modules/ops/vector_angle.py`:
	- Minor numeric/formatting cleanup; import order; multi-line norm computation for readability.
	- Effects: Behavior unchanged; style/clarity improved.

- `modules/ops/transformation.py`:
	- Add annotations; friendlier error messages; expose helpers with typed signatures:
		`get_rotation_translation_from_transform`, `get_transform_from_rotation_translation`,
		`inverse_transform`, `skew_symmetric_matrix`, `rodrigues_*`.
	- Effects: More discoverable ops; easier static analysis and reuse.

- `modules/ops/radius_search.py`:
	- New torch/NumPy input coercion and CPU staging for C++ op; optional `out_device` to move outputs.
	- Doc clarifies shapes: neighbor indices length equals `neighbor_limit` if >0.
	- Effects: Interop with NumPy datasets; explicit device transfers; fewer hidden CPU/GPU surprises.

- `modules/ops/grid_subsample.py`:
	- Similar wrapper upgrades as `radius_search`: typed signature, input coercion, `out_device` support.
	- Effects: Predictable device handling for subsampled outputs.

- `modules/ops/pointcloud_partition.py`:
	- Fix top-k on sparse inputs: cap k by available points; use sentinel + masks; pre-allocate (M, K).
	- Replace hard `.cuda()` with device-aware tensors; comments/docstrings updated.
	- Effects: No index errors on sparse nodes; fixed-shape outputs with valid mask semantics.

- `transforms/functional.py`:
	- Torch-first transforms with optional NumPy I/O; seedable randomness; device-aware ops.
	- New helpers: `_to_torch`, `_maybe_numpy`; expanded set (normalize, sample/random_sample, jitter,
		shuffle, dropout, rescale, rotate-along-up, crop-by-plane/point, sample plane/viewpoint).
	- Each function supports `device`, optional `seed`, and `output_numpy` flag.
	- Effects: Deterministic, GPU-friendly data transforms; consistent return types and shapes.

- `utils/common.py`:
	- String/format cleanup; add `best_torch_device()` utility (CUDA→MPS→CPU).
	- Effects: Simplifies choosing a sensible default device.

- `utils/data.py`:
	- Precompute pipeline is torch-first; C++ CPU ops wrapped with explicit CPU staging then moved to device.
	- Collate functions accept `device` and return tensors on that device; normals handled consistently.
	- Add `calibrate_neighbors_stack_mode` returning torch LongTensor; `validate_dataset` helper.
	- Effects: Cleaner dataloading on CPU/GPU, easier neighbor-limit calibration, improved robustness.

- `utils/pointcloud.py`:
	- Torch-first rework: nearest neighbors via `torch.cdist` with KDTree fallback; transform compose/inverse
		on tensors; uniform rotation samplers (quaternions), quaternion→matrix conversion.
	- New utilities to ensure/pad exact point counts; device-aware conversions; normals regularization in torch.
	- Effects: Faster GPU paths; deterministic utilities; easier integration with training code.

- `utils/registration.py`:
	- Torch-aware wrappers for RRE/RTE and registration errors; preserve SciPy semantics for anisotropic MSE/MAE.
	- Keep numpy returns for some metrics for backward compatibility but accept torch inputs.
	- Effects: Flexibility in pipelines mixing torch/numpy while improving device consistency.

- `utils/torch.py`:
	- Minor formatting; import order; doc-friendly signatures; same behavior.
	- Effects: No functional changes; readability.

- `utils/visualization.py`:
	- Type hints; robust numpy conversion helpers; guard TSNE normalization; window title parameters.
	- New builders to prepare KNN/node visualizations; minor API changes for correspondence drawing.
	- Effects: More robust visualization with mixed torch/numpy inputs; fewer edge-case crashes.

## Change Types
- All modified (`M`); no new or deleted files in this range.

## Notes
- Introduces type hints and updates top-k logic for sparse point clouds.
- Adjustments to avoid detach issues where expected tensors may be lists.
- Significant updates across utils and registration modules; heavy changes in point cloud utilities and visualization.

---
Generated automatically from `git diff --stat`, `--name-status`, and `--numstat` for range `HEAD~1..HEAD` at time of generation.