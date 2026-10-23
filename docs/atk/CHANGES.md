# ATK patch 4 — 2026-10-23

Server: `examples/diffusion-gemma/diffusion-gemma-server.cpp`; one library addition in
`src/llama-context.cpp` (`llama_diffusion_self_cond_clear`). Build marker in `/props` →
`build_info`: `+ ATK patch 4`; `/props.atk.patch` = 4, with `features`, `host_self_cond` and
`host_self_cond_features` so a client can check what it may ask for.

Every item is opt-in. With no new request fields the sampler runs patch 3 exactly, with ONE
exception, `demolish_clear_sc` (on by default — a fix to patch 3's plateau re-noise, below).
Nothing here touches CUDA code: the new behaviour lives in the host sampling path and in one
generic backend call, so a request that uses a patch-4 feature runs that path on EVERY block
(patch 3's features ran on the read-out block only, because only that block left the device loop).
Requests without patch-4 fields keep the device loop where they had it.

## Launch flags now honoured (server and CLI)

The PR's server and CLI built the loader/context params by hand and dropped most of what
`common_params_parse` accepted. Both now start from `common_model_params_to_llama` /
`common_context_params_to_llama`, so these apply: `-fa`, `-ctk` / `-ctv` (KV cache
quantization — VRAM freed here is VRAM for experts), `--override-kv` (e.g.
`--override-kv diffusion-gemma.expert_used_count=int:6` to bench a cheaper routing), `--mlock`,
`--no-mmap`, `-sm` / `-ts` / `-mg`, `--no-kv-offload`, `--check-tensors`. Pinned afterwards:
the batch is the canvas, and `swa_full = true` (common's default is the window-size cache; the
prefix cache's partial `llama_memory_seq_rm` is exact only on a full-size cache, which was
llama's own default the PR relied on). Without `-ngl`, every layer is still offloaded (999).

## "Host self-cond" (lab mode)

Features that read or edit the per-position top-k distribution need the server launched with
`--no-diffusion-device-selfcond`. The default (device self-cond, fused soft embedding) never
exposes the distributions to the host, so a request asking for one of these on a default
server is refused with a message naming the flag. Cost of lab mode: the non-fused gather graph
and a ~512 KB copy each way per step — about a millisecond against a 26B forward. Lab mode also
disables the device loop for every request. Marked **[host]** below.

## Request fields

`atk_sampler` (all optional):

| field | default | what it does |
|---|---|---|
| `canvas_length` (also `atk_canvas.length`) | server's | shorter canvas, 8..server canvas. Step cost is linear in canvas tokens; the model was trained on 256 only — an EXPERIMENT, bench before relying on it. `max_tokens` buys more, shorter blocks. |
| `stability_steps` | 1 | argmax unchanged for this many consecutive steps to stop (vLLM uses "a couple") |
| `eog_tail` | false | once an EOG is the argmax at the same position for `eog_tail_stable` steps, positions after it leave the stop test, the acceptance budget and the plateau's demolition set |
| `eog_tail_stable` | 2 | |
| `eog_tail_pad` | false | ... and are written as the pad token each step (a model-INPUT change: the block then looks like a finished, padded SFT block) |
| `noise_text` | false | re-noise draws from text tokens only (NORMAL / USER_DEFINED / BYTE), not from control and `<unused…>` ids. `/props.atk.noise_text_tokens` is the table size. |
| `noise_clear_sc` | false | a re-noised position also forgets its self-conditioning |
| `demolish_clear_sc` | **true** | a plateau-demolished position forgets its self-conditioning. Patch 3 gave it pure noise but left its memory, so the model was still told what it had thought there. `false` = patch 3. |
| `sprint_below` | 0 (off) | when the mean entropy is already below this, skip `sprint_jump` schedule steps (only temperature and top-k annealing depend on the schedule), never below `cur_step = sprint_min` |
| `sprint_jump` / `sprint_min` | 2 / 2 | |
| `pos_bias` | 0 | acceptance ORDER key = entropy × (1 + pos_bias × i/L): positive commits the left first; the BUDGET still sums raw entropy |
| `accept_order` | `"entropy"` | `"margin"` **[host]**: order by top-1 − top-2 probability instead |
| `bound_end` | −1 (constant) | entropy budget scheduled linearly from `entropy_bound` (first step) to `bound_end` (last) |
| `temp_pos_gain` | 0 | **[host]** per-position temperature: T_i = T × (1 + g·z_i), z_i ∈ [−1, 1] the entropy rank (noisiest = +1). Changes how a position is sampled, not what is fed back. |
| `rep_penalty` | 0 | **[host]** spatial repetition penalty (plan item 6): p_i(v) ×= exp(−η Σ_j w(\|i−j\|) p_j(v)) over ±`rep_window`, w(d) = exp(−(d−1)/`rep_decay`), for tokens with ≥ `rep_min_chars` characters including a letter/digit. Applied to the sample, the argmax canvas and the fed-back distribution. |
| `rep_window` / `rep_decay` / `rep_min_chars` | 8 / 4.0 / 3 | |
| `sc_sharpen` | 1 | **[host]** fed-back distribution raised to this power and renormalised (> 1 commits, < 1 explores) |
| `lateral_noise` | 0 | **[host]** a re-noised position draws from a neighbour's top-k with this probability (tried before warm noise) |
| `osc_steps` | 0 (off) | oscillation hold (cheap form of plan item 5): a position alternating A,B,A,B for this many steps is oscillating; when 1..`osc_max_pos` positions oscillate and the stop test has not passed, each is pinned for `osc_hold_steps` steps at its lower-entropy state |
| `osc_max_pos` / `osc_hold_steps` | 4 / 3 | |
| `n_candidates` | 1 | best-of-N: run with seeds seed, seed+1, … (prompt KV reused), return the lowest best-mean-entropy canvas; all candidates in `atk_candidates` (`seed`, `answer`, `best_entropy`, `last_entropy`, `steps`, `hit_eog`, `chosen`). Forces the readout on. |

`atk_canvas` additions:

| field | what it does |
|---|---|
| `length` | as `atk_sampler.canvas_length` |
| `soft_hold` | `[bool,…]`, one per given token **[host]**: the position starts as its token, un-noised, with its self-conditioning seeded to that token at `soft_hold_p` (default 0.9) — "prefer to keep this, but you may fix it". A hard `hold` fixes the token; a soft hold only remembers it. |
| `self_cond` | `{"k": N, "ids": [[…],…], "probs": [[…],…]}` **[host]**: seed the first step's self-conditioning from a previous readout, per position. This makes denoise → verify → re-noise the failed spans (`noise_map`) → continue a loop of stateless requests in which the kept positions resume with their memory (plan item 1, forward-only). |

Top-level additions: `atk_readout_sc: N` **[host]** returns `atk_canvas_out.self_cond` (top-N per
position, the LAST step's); `atk_trajectory_text: true` adds the argmax canvas as `text` to every
trajectory row; `atk_stream_steps: true` (with `stream: true` on `/v1/completions`) sends an
`atk_step` event after every denoising step: `block`, `step`, `text`, `mean_entropy`, `accepted`,
per-position `entropy` — the denoising made visible, colourable by confidence.

## Readout (`atk_canvas_out`), new fields

`length`, `sprints`, `osc_holds`, `eog_pos` (first EOG in the returned canvas), `tail_from`
(first position the stop test stopped watching, −1 = never), `sc_cleared` (self-conditioning rows
forgotten), `host_self_cond`, `self_cond` (with `atk_readout_sc`). Trajectory rows add `sprint`,
`eog_pos`, `tail`, `osc_holds`, `bound`, and `text` with `atk_trajectory_text`.
`/v1/chat/completions` now carries `atk_canvas_out` and `atk_candidates` too (its `finish_reason`
is still always `stop`, as in the PR; `/v1/completions` reports `length` on a cut-off answer).

## Library

`llama_diffusion_self_cond_clear(ctx, positions, n)` (`include/llama.h`, `src/llama-context.cpp`):
zero the self-conditioning of the given canvas positions wherever it lives — host top-k rows, the
device's sparse top-k rows, or the fused soft-embedding columns — through
`ggml_backend_tensor_set`, no CUDA code. Returns the count cleared, or −1 when no state is live.

## Bench plan (what each item must earn)

1. `eog_tail`: one short-answer prompt with the readout; if the trajectory's `tail` positions carry
   the flips and the entropy, compare steps-to-stop with and without `eog_tail`.
2. `canvas_length` 128 and 64 against 256 on short tasks: quality first, then ms/step.
3. `--override-kv diffusion-gemma.expert_used_count=int:N` for N below the model's: ms/step and
   quality on the standard set.
4. `noise_text` vs. the reference sampler's noise: check what the reference draws from before
   deciding which is "right".
5. `demolish_clear_sc` on vs. off on prompts that plateau.
6. Everything **[host]** needs lab mode; measure lab mode's own cost first (ms/step with and
   without `--no-diffusion-device-selfcond`, no features) so the features are judged net of it.

## Not in this patch

Stall bifurcation proper (fork the canvas; `osc_*` is the one-canvas form), stable-position KV
compaction (plan item 7; needs a bench result first), and anything in the device loop — the loop
runs the PR's algorithm exactly.

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
