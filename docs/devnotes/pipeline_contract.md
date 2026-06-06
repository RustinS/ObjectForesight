Pipeline Contract (Train → Infer → Viz)
=======================================

Batch contract (training/inference)
-----------------------------------
- Required keys (per item unless noted):
  - `scene_pcd`: float32 tensor/ndarray, shape [N,3] (batched to [B,N,3] by collate)
  - `init_pose`: dict with:
    - `t0`: float32, shape [3]
    - `rot6d0`: float32, shape [6]
  - `target_future`: float32, shape [H,9] or [B,H,9]
  - Context sequence (P frames, required):
    - `context_len`: int P
    - `context_frame_ids`: int32, shape [P]
    - `context_T_cam_anchor_obj`: float32, shape [P,4,4]
    - `context_init_9d`: float32, shape [P,9]
    - `context_bbox_norm`: float32, shape [P,4]
  - Canonicalization/camera meta (window of length P+H):
    - `frame_ids`: int32, shape [P+H]
    - `T_c_w`: float32, shape [P+H,4,4]  (camera<-world if `extrinsics_convention=="c2w"`)
    - `T_c_o`: float32, shape [P+H,4,4]  (camera<-object, FoundationPose track)
    - `T_cam_anchor_obj`: float32, shape [P+H,4,4] (object poses re-expressed in the dataset anchor camera)
    - `anchor_mode`: str, usually `window_start`
    - `anchor_frame_idx`: int (global)
    - `anchor_local_idx`: int (window-local index of dataset anchor)
    - `extrinsics_convention`: `"c2w"` or `"w2c"`
  - Object info (optional, used for context features/overlays):
    - `object.bbox_norm`: float32, shape [4]
    - `object.bbox`: float32, shape [4]
    - `mesh_path`: str or None

Token format
------------
- 9D tokens in strict order: `[t_x, t_y, t_z, rot6d(6)]`
- Rotation is 6D; projection to SO(3) via `rot6d_to_matrix`
- Model inputs/outputs: `(B,H,9)`

Temporal contract and indices
-----------------------------
- `P`: context length (`context_len`)
- `H`: prediction horizon (`cfg.data.H`)
- Dataset anchor (training perspective): `aLoc = sample['anchor_local_idx']`
- Visualization/index anchor (last context): `anchor_idx = aLoc + (P - 1)`
- First predicted frame index (window-local): `start = anchor_idx + 1`
- Predicted range: `[start, stop)` where `stop = start + Hn`
- Ground-truth slice used for metrics/overlays: `T_cam_anchor_obj[start:stop]` with shape `[Hn,4,4]`

Normalization/denormalization ownership
---------------------------------------
- Single source of truth: `geom.canonicalize.canonicalize_preds_to_anchor(...)`
  - Call with `do_denorm=True`
  - Pass `t_mean=[0,0,0]`, `t_std=[t_scale, t_scale, t_scale]`
  - Inference threads `t_scale` from the model (e.g., `PoserV1._scale_t`) into `pred_meta*.json`
  - For current DiT/AR implementations, translations are metric; default `t_scale=1.0`
- Neither model nor viz/infer should re-scale translations outside this call (prevents double scaling)

Canonicalization/camera convention
----------------------------------
- Function: `canonicalize_preds_to_anchor(pred_9d, meta, pred_mode, extrinsics_convention, do_denorm=True)`
  - Inputs:
    - `pred_9d`: `(B,H,9)` tokens
    - `pred_mode`: `"abs_in_anchor_cam"`, `"deltas_from_prev_cam"`, `"deltas_from_anchor_cam"`, or `"relative_world_from_o0"`
    - `extrinsics_convention`: `"w<-c"` if `c2w`, or `"c<-w"` if `w2c`
    - `meta` includes: `K`, `frame_ids`, `anchor_frame_idx`, `anchor_local_idx`, `T_c_w`, `T_c_o`, `T_cam_anchor_obj`, `t_mean`, `t_std`
  - Output: `T_camA_obj_pred` with shape `(B,H,4,4)` absolute poses in the anchor camera
- Overlays and per-frame projections:
  - If images are resized, scale intrinsics with `scale_intrinsics` to the target image size before projection
  - Map camera frames with convention:
    - `"c2w"` (arrow `"w<-c"`): `T_wc_k = invert(T_c_w[k])`, `T_ck_ca = T_wc_k @ T_cw_anchor`
    - `"w2c"` (arrow `"c<-w"`): `T_ck_ca = T_c_w[k] @ invert(T_c_w[anchor])`

Train vs. Infer vs. Viz usage
-----------------------------
- Training (`train_main.py`):
  - Uses dataset anchor `aLoc` for canonicalization within metrics
  - Slices GT with `aLoc + P : aLoc + P + Hn`
- Inference (`infer_main.py`):
  - Uses `aLoc` when canonicalizing for metrics
  - Index math for slicing/logs:
    - `anchor_idx = aLoc + (P - 1)`
    - `start = anchor_idx + 1`, `stop = start + Hn`
  - Saves `pred_meta*.json` with: frames, `t_scale`, `ctx_len`, `anchor_idx`, `[start, stop)`, `H`, `pred_mode`
- Visualization (`viz_main.py`):
  - Build `sample_v` with:
    - `sample_v['anchor_local_idx'] = anchor_idx`
    - `sample_v['anchor_frame_idx'] = frames[anchor_idx]` when frames exist
  - Canonicalize predictions via `canonicalize_preds_to_anchor(..., do_denorm=True)` using `t_scale` from `pred_meta*.json`
  - Slice GT identically: `T_cam_anchor_obj[start:stop]`

Assertions and tests
--------------------
- Index math invariants asserted in inference/viz:
  - OOB and shape checks on GT slicing
  - Identity canonicalization: feeding GT tokens through pred path yields `canon_id_rot_deg_mean < 1e-3`
- Synthetic tests cover:
  - Anchor/index math
  - Canonicalization identity
  - Viz slicing parity with infer


