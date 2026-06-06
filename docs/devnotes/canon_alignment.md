Canon/Index/Scale Alignment (Train • Infer • Viz)
=================================================

Anchor semantics
----------------
- context_len = P
- Dataset anchor (training perspective): `aLoc = sample['anchor_local_idx']`
- Visualization/indices anchor (last context): `anchor_idx = aLoc + (P - 1)`
- Prediction slice: 
  - `start = anchor_idx + 1`
  - `stop  = start + Hn`
  - Ground-truth slice used for metrics/overlays: `T_cam_anchor_obj[start:stop]` (shape `[Hn,4,4]`)

Token format
------------
- Tokens are 9D in the exact order: `[t_x, t_y, t_z, r6d_0..5]`
- Rotation parameterization is 6D (`rot6d`); projection to SO(3) via `rot6d_to_matrix`
- Canonicalizer expects `(B,H,9)` and produces absolute `T_camA_obj` in the anchor camera

Normalization ownership
-----------------------
- One and only one place applies translation de-normalization: `canonicalize_preds_to_anchor(...)`
- Call with `do_denorm=True`, and pass:
  - `t_mean = [0,0,0]`
  - `t_std  = [t_scale, t_scale, t_scale]`
- Thread `t_scale` from the model via inference meta (`pred_meta*.json`). For current DiT/AR implementations, tokens are already metric; set `t_scale=1.0`. If a future model emits normalized translations, set `t_scale` to the training-time scale or call `do_denorm=False`.

Training-time buffers (model-owned)
-----------------------------------
- `_scale_t` (float tensor): nominal translation scale used for token normalization (if enabled)
- `_scale_r` (float tensor): nominal rot6d scale (if enabled)
- `_dnorm_means` / `_dnorm_scales` (9-D): per-channel means/stds for depth-parameterized tokens `[u,v,s,r6]` (DiT)
- `_dnorm_fitted` (uint8): whether d-norm statistics have been fit
- `_T_train` (int tensor): diffusion horizon `T` used during training (DiT)

Owner of de-normalization
-------------------------
- The canonicalizer (`canonicalize_preds_to_anchor`) exclusively performs de-normalization of translations using `t_mean` and `t_std`
- Neither the model nor viz/infer should re-scale tokens outside this call

Canonicalization contract
-------------------------
Function: `canonicalize_preds_to_anchor(pred_9d, meta, pred_mode, extrinsics_convention, do_denorm=True)`

Inputs:
- `pred_9d`: `(B,H,9)` tokens in `[t_xyz, rot6d]`
- `pred_mode`: one of `abs_in_anchor_cam`, `deltas_from_prev_cam`, `deltas_from_anchor_cam`, `relative_world_from_o0`
- `extrinsics_convention`: `"w<-c"` (c2w) or `"c<-w"` (w2c)
- `meta` dict includes:
  - `K`, `frame_ids`
  - `anchor_frame_idx` (global int), `anchor_local_idx` (window-local int)
  - `T_c_w`, `T_c_o`, `T_cam_anchor_obj`
  - `t_mean=[0,0,0]`, `t_std=[t_scale, t_scale, t_scale]`

Outputs:
- `T_camA_obj_pred`: `(B,H,4,4)` absolute poses in the anchor camera
- Optional `info` dict with parity diagnostics (projection signature, hygiene checks)

Train vs. Infer vs. Viz index usage
-----------------------------------
- Training (metrics inside `train_main.py`):
  - Uses dataset anchor (`aLoc`)
  - Slices GT with `aLoc + P : aLoc + P + Hn`
- Inference (`infer_main.py`):
  - For metrics canonicalization, pass `anchor_local_idx = aLoc` (training perspective)
  - Use indices for slicing/logs:
    - `anchor_idx = aLoc + (P - 1)`
    - `start = anchor_idx + 1`, `stop = start + Hn`
  - Write `pred_meta*.json` with `t_scale` threaded (see above)
- Visualization (`viz_main.py`):
  - Build `sample_v` with:
    - `sample_v['anchor_local_idx'] = int(anchor_idx)` (last context for viz)
    - `sample_v['anchor_frame_idx'] = frames[anchor_idx]` when frames exist
  - Canonicalize predictions via `canonicalize_preds_to_anchor(..., do_denorm=True)` with `t_std=t_scale`
  - Use identical `start/stop` slicing for GT as in infer

Directionality and overlays
---------------------------
- Directionality probe in viz uses `anchor_idx` (last context)
- Camera mapping for overlays (`T_ck_ca`) respects `extrinsics_convention`:
  - If c2w (`"w<-c"`): `T_wc_k = invert(T_c_w[k])`; `T_ck_ca = T_wc_k @ T_cw_anchor`
  - If w2c (`"c<-w"`): `T_ck_ca = T_c_w[k] @ invert(T_c_w[anchor_idx])`

Sanity & acceptance hooks
-------------------------
- Identity canonicalization (infer): feeding GT tokens through pred path must satisfy `canon_id_rot_deg_mean < 1e-3`
- Guard against double-scaling: no extra scaling outside canonicalizer; toggling `t_scale` should visibly affect results
- R matrix hygiene: orthogonality and determinant errors remain within `1e-6–1e-7`


