# ATK patch 3 — 2026-10-22

Server: `examples/diffusion-gemma/diffusion-gemma-server.cpp`. Build marker in
`/props` → `build_info`: `+ ATK patch 3`. Every item is opt-in or diagnostic;
with no new request fields the sampler runs the PR's algorithm exactly, except
`keep_best` (below), which is on.

## Request (all optional)

`atk_canvas`:
- `noise_map`: `[float,…]` per-position noise (0..1), overriding `noise` where
  given. Lets a client re-noise exactly the spans a verifier rejected.
- `start`: 0..1, where on the schedule to start. Without it, and with a
  `noise_map`, the schedule starts at the mean noise of the given, unheld
  positions (blank positions count as 1.0).

`atk_sampler`:
- `warm_start`, `warm_end` (0..1, default 0/0): a position not accepted this
  step keeps the model's own sample with probability rho instead of a uniform
  random token; rho runs linearly warm_start → warm_end over the schedule.
- `plateau_steps` (int, default 0 = off), `plateau_eps` (0.002),
  `plateau_frac` (0.2), `plateau_extra` (8), `plateau_max` (2): when the mean
  entropy has not fallen by `plateau_eps` for `plateau_steps` steps and the
  stop test has not passed, the `plateau_frac` noisiest unheld positions and
  their neighbours are forced out of the accepted set (pure noise) and
  `plateau_extra` steps are added, at most `plateau_max` times per canvas.
- `keep_best` (bool, default true): return the lowest-mean-entropy canvas
  seen when the last step's is worse. The swap is made in the canvas itself,
  so the readout, the choice text and the KV commit agree.
- `no_control` (bool, default false): a control token (other than EOG)
  sampled into a canvas position is never accepted; the position is
  re-noised.

## Readout (`atk_canvas_out`), new fields

- `text`: the canvas as ONE detokenized string (or `text_hex` when the bytes
  are not valid UTF-8 — a canvas can end inside a character).
- `offsets`: canvas_length + 1 byte offsets into `text`; token i's text is
  `text[offsets[i]:offsets[i+1]]`. Clients slice this instead of joining
  `pieces_hex`.
- `control`: per position, whether it holds a control token (not EOG). Such
  a token renders as an EMPTY piece — the text either side of it is glued
  together. This is the suspected mechanism of ATK's eaten-character defect
  (`from.models`, `SHIELDCTORS`); `control_count` says how many the returned
  canvas holds.
- `trajectory`: one row per step — `step`, `cur_step`, `mean_entropy`,
  `accepted`, `flips`, `control`, `temp`, `ms`, `plateau`.
- `plateaus`, `from_best`, `best_entropy`, `last_entropy`.

## Not in this patch (plan items 5–7)

Stall bifurcation, the spatial repetition penalty (host sampling paths only
would be reachable; the GPU path is the default) and stable-position KV
compaction. Each needs a bench result first; see `DIFFUSION_ENGINE_PLAN.md`.
