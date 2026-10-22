# ATK's diffusion branch — `atk/diffusion-gemma`

This branch is ATK's diffusion inference engine. It is NOT a mirror of
upstream: `master` in this fork tracks `ggml-org/llama.cpp` for reference,
and ATK's autoregressive models keep running on upstream llama.cpp (through
llama-cpp-python). Only diffusion models run from this branch.

History:

1. `dd0cf044` — PR #24427 (diffusion-gemma block diffusion), the commit ATK
   was tested against on 2026-10-09 and built on 2026-10-19.
2. ATK patch 2 — canvas layout, held positions, readout, per-request sampler
   constants, prefix KV reuse, `-ot` passthrough. Previously a `.patch` file
   in the ATK repo applied by `get_diffusion_gemma.bat`; now history.
3. ATK patch 3 — see `DIFFUSION_ENGINE_PLAN.md` for the design and
   `CHANGES.md` for what each commit added.

Moving the base: never rebase onto a moving `master`. Pin the base, move it
on purpose, with the diff in front of you, and rebuild + bench before the
pin changes in `get_diffusion_gemma.bat`.
