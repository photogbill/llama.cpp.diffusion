# ATK diffusion engine — plan of record

*2026-10-22. Written against the code as it stands: upstream
`examples/diffusion/diffusion.cpp` (406 lines, LLaDA/Dream/RND1) and PR
#24427's `diffusion-gemma-server.cpp` (1,518 lines) with ATK patch 2. Every
claim about what the sampler does was read from those files, not from a
paper. The fork is `photogbill/llama.cpp.diffusion`; the diffusion-specific
code it will own is ~3,000 lines on top of a pinned, vendored substrate.*

## What DiffusionGemma's sampler actually does (read, not assumed)

1. The canvas starts as **uniformly random tokens** — `rand_tok(rng)` — not
   `[MASK]`. This is uniform-noise discrete diffusion, not masked diffusion.
2. Every step runs one bidirectional decode over all 256 canvas positions
   and samples every position (temperature annealed 0.8 → 0.4 across the
   schedule). Positions are sorted by entropy; samples are **accepted** in
   ascending order while the running entropy total stays under
   `entropy_bound` (0.1). Every position NOT accepted this step is
   **re-randomised**. Acceptance is recomputed from scratch each step:
   **no token is ever locked.**
3. The memory across steps is **self-conditioning**: the top-`SC_K` (256)
   token probabilities per position from the previous step are fed back
   as a soft input embedding (the PR's "approximation of soft
   embeddings"; tensors `self_cond_down/gate/up/norm`). This is trained.
4. Stop: `argmax` unchanged from the previous step AND mean entropy over
   free positions `< stop_entropy` (0.005). A block's output is the
   argmax canvas, which is then **committed into the KV** as prefix for
   the next 256-token block (block diffusion). The prompt prefix is in KV
   once; the canvas positions are decoded and removed every step.
5. ATK patch 2 adds: held positions put back after every step; a laid-out
   first canvas with a noise fraction that sets the starting step;
   per-position entropy and argmax-flip counts read out; prefix KV reuse
   across requests; per-request sampler constants.

Consequences for every idea below: there is no "frozen token" to cache,
only *stable* ones (accepted with the same argmax for k steps running);
the simplex-relaxation idea is already the architecture; and anything
that wants to "unlock" a token has nothing to unlock — it wants to
**re-noise** it, which patch 2 can already do at request start.

## Build list, ranked by value per line

### 1. Verifier-guided re-noise (the ATK-specific one)
Server: `atk_renoise` on a running or just-finished request — a list of
positions (or byte ranges, mapped by the server) to re-randomise, a
`noise` fraction, and `steps` to continue, **keeping self-conditioning
for every other position**. Client side the verifier is anything: a
Python parser (Cognitive Coder), a GBNF grammar (JSON extraction — the
diffusion server has no grammar sampling, this gives it one), a syllable /
rhyme / stress scorer (lyrics). The loop is: denoise → check → re-noise
the spans that fail → continue. This is the forward-only, no-gradient
version of "verifier in the loop", and nothing upstream has it.

### 2. Canvas text as one string, with offsets
The character-eating defect (`from.models`, `SHIELDCTORS`): the readout
returns per-token pieces that the client joins. Return ONE detokenized
string plus per-position byte offsets. Pin with a round-trip test of
`from .models import x` and `if 'a' in b:`.

### 3. Warm noise
Re-randomised positions draw from `uniform` today. Add a `noise_source`:
`uniform | self_cond | mix(ρ)` — a re-noised position draws from its own
previous top-K distribution with probability ρ, uniform otherwise.
Schedule ρ from 0 (early, explore) to ~0.7 (late, settle). This is what
"correlated noise across steps" means for a uniform-token sampler, and it
is a 30-line change in the acceptance loop. Measure on the bench.

### 4. Best-checkpoint retention + plateau re-noise (sawtooth)
Keep the canvas with the lowest mean entropy seen in the run. On a
plateau (mean entropy not falling for `k` steps, not yet confident),
re-noise the top 20 % highest-entropy positions ± 2 neighbours and run
`n` more steps; keep the result only if mean entropy improved. All inside
the sampler loop; no model change.

### 5. Stall bifurcation
`line_flips` already counts argmax oscillation. When a few positions
alternate between two argmaxes while the rest is stable, fork the canvas
in two with each alternative held, run 3 steps each, keep the lower
mean-entropy branch. Costs 2× for 3 steps; the alternative is 20 steps of
oscillation.

### 6. Spatial repetition penalty
Before sampling position `i`, subtract `η · Σ_j w(|i−j|) · p_j(v)` for
non-stopword `v` over a ±8 window, `w` exponential. The diffusion
analogue of llama.cpp's repetition penalty; dLLM stutter is a known mode.

### 7. Stable-position compaction (approximate, measure first)
Positions accepted with the same argmax for `k` steps: reuse their K/V
from the previous step instead of recomputing (Fast-dLLM's approximate
cache). Exact only at the first attention layer — above it every K/V
depends on every other position — so this is a measured approximation,
switched off until the bench says the error is below the noise.

### 8. Trajectory telemetry
Per step: mean entropy, accepted count, flips, temperature, time. Already
half there in patch 2. Return it whole for the Model Lab, and render the
per-line confidence in every ATK surface that shows diffusion output — the
readout is the one thing an AR model cannot give, and it should be seen.

## Ideas that need a trained model, not a serving trick
Layer-band looping ("theta-gamma"), untrained residual highways between
steps, RoPE wavelength compression, attention-temperature flattening,
attention-head gating by "frequency". Each is a hook that changes
activation statistics the weights never saw. Testable one knob at a time
in Athanor as experiments; expected result is degradation. The trained
version of "carry state across steps" is the model's own self-
conditioning, and the knob to turn is `SC_K` (and sampled-vs-argmax feed).

## Ideas that do not fit this engine or this card
Backward passes through a 26B at inference (the Theory-of-Mind Jacobian):
ggml has no practical inference-time backprop at this scale on 16 GB. The
forward-only cousin is classifier-free guidance, which the PR already has
(`cfg_scale`) and which can be aimed at a masked copy of the prompt's
constraints. Four parallel islands: the weights are shared, but logits
are `canvases × 256 × 262,144 × 4 bytes` — 1 GB for four — on a card the
model already fills; two is plausible, four is not. Step-annealed /
bit-plane quantization: GGUF k-quants are not bit-plane decomposable and
two models do not fit. Sequential seeds against the cached prefix get the
same diversity for free.

## Order of work
2 (the defect), 1 (the feature), 8 (so everything after it is measured),
3, 4, 5, 6, then 7 behind a bench result. Branch `atk/diffusion-gemma`:
the PR at its pinned commit as real history, patch 2 as commits, the
getter pointed at the fork.

## Status (2026-10-23, after patch 4)
- 2, 3, 4, 8: done in patch 3.
- 1: done in patch 4 as the stateless form — `atk_readout_sc` returns the
  self-conditioning, `atk_canvas.self_cond` seeds it, `noise_map` re-noises
  the rejected spans (lab mode).
- 5: the one-canvas form (`osc_*`, oscillation hold) in patch 4; the forking
  form still needs the step body refactored into a function.
- 6: done in patch 4 (`rep_*`, lab mode).
- 7: not started; needs a bench result first.
- Added in patch 4 beyond this list: launch-flag passthrough, per-request
  canvas length, post-EOG tail, text-only noise, self-cond forgetting,
  sprint, acceptance order/budget schedules, per-position temperature,
  self-cond sharpening, lateral noise, best-of-N, soft holds, per-step
  streaming. Every one is opt-in and documented in `CHANGES.md` with the
  bench it must earn.
