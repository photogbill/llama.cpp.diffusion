// OpenAI-compatible HTTP server for the block-diffusion models (diffusion-gemma).
//
// This is the llama-server analogue for the diffusion family: it loads a block-diffusion model
// once and serves the same denoising generation loop as diffusion-gemma-cli over HTTP, exposing
// the OpenAI endpoints (/v1/chat/completions, /v1/completions, /v1/models) plus the llama-server
// observability surface (/health, /v1/health, /props, /metrics, /slots) with the same response
// `timings`/`usage` objects, the same per-request timing logs, the same access log, and a Prometheus
// /metrics endpoint -- all with the metric semantics re-mapped to block diffusion (canvas blocks,
// denoising steps, s/step) instead of autoregressive token-by-token decode.
//
// Generation is NOT autoregressive: each request denoises one or more 256-token canvases against a
// cached prompt prefix (see diffusion-gemma-cli.cpp for the full description). A single llama_context
// is reused across requests and is not thread-safe, so generation is serialized behind a mutex (one
// slot); the HTTP layer still accepts many connections concurrently.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

// llama-server-style log macros (server-common.h): "srv  <func>: ..." / "slot <func>: id N | ..."
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SLT_INF(id, fmt, ...) LOG_INF("slot %12.*s: id %2d | " fmt, 12, __func__, (id), __VA_ARGS__)

// reference defaults from generation_config.json / DiffusionGemmaGenerationConfig (mirror the CLI)
static constexpr int   DEF_CANVAS_LENGTH    = 256;
static constexpr int   DEF_MAX_DENOISE_STEPS = 48;
static constexpr float ENTROPY_BOUND        = 0.1f;
static constexpr float TEMP_MIN             = 0.4f;
static constexpr float TEMP_MAX             = 0.8f;
static constexpr float CONFIDENCE_THRESHOLD = 0.005f;
static constexpr int   STABILITY_THRESHOLD  = 1;
static constexpr int   DEF_SC_K             = 256;
static constexpr int   GPU_SAMPLING_MAX_TOP_K = 1024; // CUDA diffusion sampler limit

#ifdef GGML_USE_CUDA
struct diffusion_cuda_host_pin {
    diffusion_cuda_host_pin(void * ptr, size_t size, bool enabled) : ptr(ptr), size(size) {
        registered = enabled && ptr && size && ggml_backend_cuda_register_host_buffer(ptr, size);
    }

    ~diffusion_cuda_host_pin() {
        if (registered) {
            ggml_backend_cuda_unregister_host_buffer(ptr);
        }
    }

    diffusion_cuda_host_pin(const diffusion_cuda_host_pin &) = delete;
    diffusion_cuda_host_pin & operator=(const diffusion_cuda_host_pin &) = delete;

    void * ptr = nullptr;
    size_t size = 0;
    bool registered = false;
};
#else
struct diffusion_cuda_host_pin {
    diffusion_cuda_host_pin(void *, size_t, bool) {}
};
#endif

static int diffusion_self_cond_top_k(const common_params & params) {
    const int k = params.diffusion.self_cond_top_k;
    if (k <= 0) {
        return DEF_SC_K;
    }
    return std::min(k, DEF_SC_K);
}

static void diffusion_set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void diffusion_configure_cuda_mmq_env(const common_params_diffusion & params) {
    if (params.cuda_mmq_max_x < 0) {
        return;
    }

    diffusion_set_env("GGML_CUDA_MMQ_MAX_X", std::to_string(params.cuda_mmq_max_x));
    SRV_INF("GGML_CUDA_MMQ_MAX_X=%d\n", params.cuda_mmq_max_x);
}

// ------------------------------------------------------------------------------------------------
// per-request generation parameters and result
// ------------------------------------------------------------------------------------------------
struct diffusion_request {
    int      canvas_length = DEF_CANVAS_LENGTH;
    int      n_steps       = DEF_MAX_DENOISE_STEPS;
    int      max_canvases  = 1;     // ceil(max_tokens / canvas_length)
    int      topk_fixed    = 0;     // 0 = full softmax
    int      topk_start    = 0;
    int      topk_end      = 0;
    bool     topk_tail     = false;
    bool     ignore_eos    = false; // run all max_canvases blocks (don't stop at end-of-text)
    uint32_t seed          = 1234;

    // ATK patch 2: the sampler's reference constants, overridable per request ("atk_sampler").
    float    entropy_bound = ENTROPY_BOUND;        // uncertainty committed per step (Algorithm 1, b)
    float    stop_entropy  = CONFIDENCE_THRESHOLD; // adaptive stopping: mean entropy below this ...
    float    temp_max      = TEMP_MAX;             // ... temperature annealed from temp_max ...
    float    temp_min      = TEMP_MIN;             // ... to temp_min across the schedule

    // ATK patch 2: the FIRST canvas laid out by the client ("atk_canvas"). Positions with a token
    // >= 0 start as that token; -1 starts random, as every position does without this. Held
    // positions are put back after every step, so the model writes only the rest, seeing the held
    // text on both sides. `canvas_noise` scrambles that fraction of the given, unheld tokens before
    // the first step, and the schedule starts at that point of the noise path (t = noise): 1.0 is
    // "from noise", 0.3 is "this draft, lightly re-noised".
    bool                     has_canvas   = false;
    std::vector<llama_token> canvas_init;
    std::vector<char>        canvas_hold;
    float                    canvas_noise = 1.0f;
    // ATK patch 2: return the first canvas as tokens, text pieces, final per-position entropy and
    // how often each position's best guess changed ("atk_readout"; always on with atk_canvas).
    bool                     readout      = false;
    // ATK patch 2: keep the KV of the longest prompt prefix shared with the previous request.
    bool                     cache_prompt = true;

    // ATK patch 3: per-position noise for the laid-out canvas ("atk_canvas.noise_map"), overriding
    // `canvas_noise` where given, so a client can re-noise exactly the spans a verifier rejected
    // and leave the rest as the model's own draft; and where on the schedule to start
    // ("atk_canvas.start", 0..1). Without `start` the schedule starts at the MEAN noise of the
    // given, unheld positions (blank positions count as 1.0).
    std::vector<float>       canvas_noise_map;
    float                    canvas_start = -1.0f;
    // ATK patch 3: warm noise ("atk_sampler.warm_start" / "warm_end"). A position not accepted
    // this step is re-randomised; with probability rho it keeps the model's own sample for that
    // position instead of a uniform random token. rho runs linearly from warm_start at the first
    // step to warm_end at the last. 0/0 (the default) is the PR's behaviour exactly.
    float    warm_start = 0.0f;
    float    warm_end   = 0.0f;
    // ATK patch 3: plateau re-noise ("atk_sampler.plateau_*"). When the mean entropy has not
    // fallen by plateau_eps for plateau_steps consecutive steps, and the stop test has not
    // passed, the plateau_frac highest-entropy unheld positions and their neighbours are forced
    // out of the accepted set (uniform noise, not warm) and plateau_extra steps are added to the
    // schedule, at most plateau_max times per canvas. plateau_steps 0 (the default) = off.
    int      plateau_steps = 0;
    float    plateau_eps   = 0.002f;
    float    plateau_frac  = 0.2f;
    int      plateau_extra = 8;
    int      plateau_max   = 2;
    // ATK patch 3: keep the lowest-mean-entropy canvas seen and return it when the last step's
    // canvas is worse ("atk_sampler.keep_best", default on). Off = the PR's behaviour.
    bool     keep_best     = true;
    // ATK patch 3: refuse a control token (other than end-of-generation) sampled into a canvas
    // position ("atk_sampler.no_control"): it is never accepted, so the position is re-noised.
    // Off by default; the readout ALWAYS reports how many control tokens the canvas holds.
    bool     no_control    = false;

    // ---- ATK patch 4 ---------------------------------------------------------------------------
    // Every field below is opt-in; the defaults reproduce patch 3 exactly, except demolish_clear_sc
    // (a fix to patch 3's plateau re-noise, see below). "host self-cond" marks features that read or
    // edit the per-position top-k distribution on the host: they need the server launched with
    // --no-diffusion-device-selfcond (reported as "host_self_cond" in /props); a request that asks
    // for one on a device-self-cond server is refused with a message naming the flag.

    // "atk_canvas.length": a shorter canvas for this request (<= the server's canvas, which is the
    // model's trained length: 256). The step cost is linear in canvas tokens, so this is the one
    // knob that cuts the per-step floor for short answers -- but the model never saw a shorter
    // block in training, so it is an EXPERIMENT: bench it before relying on it. 0 = server's.
    int      canvas_len_req = 0;
    // "atk_sampler.stability_steps": the argmax must be unchanged for this many consecutive steps
    // (over the positions the stop test watches) for the stop test to pass. The PR uses 1.
    int      stability_steps = 1;
    // "atk_sampler.eog_tail": once an end-of-generation token has been the argmax at the same
    // position for eog_tail_stable steps, the positions AFTER it are the tail of a finished
    // answer. They leave the stop test (mean entropy and stability) and the plateau's demolition
    // set, so a flickering position in the tail no longer keeps a 12-token answer running for
    // the full schedule. eog_tail_pad additionally writes the pad token into the tail each step
    // (a model-INPUT change: the model then sees a padded block, as a finished SFT block looks).
    bool     eog_tail        = false;
    int      eog_tail_stable = 2;
    bool     eog_tail_pad    = false;
    // "atk_sampler.noise_text": re-noise draws from TEXT tokens only (NORMAL / USER_DEFINED / BYTE
    // attributes), not from the control and <unused...> ids that make up a few percent of a Gemma
    // vocabulary. The PR (and the device loop) draw uniformly over all 262,144 ids.
    bool     noise_text      = false;
    // "atk_sampler.noise_clear_sc": a re-noised position also forgets its self-conditioning (the
    // model gets no memory of what it thought there). Off = the PR's behaviour: noise in the
    // token, memory intact. "demolish_clear_sc" (default ON) does the same for the positions a
    // plateau demolishes -- patch 3 gave them pure noise but left their memory in place, so the
    // model was still told what it had thought there; this completes the demolition.
    bool     noise_clear_sc    = false;
    bool     demolish_clear_sc = true;
    // "atk_sampler.sprint_*": the mirror of the plateau. When the mean entropy is already below
    // sprint_below, skip sprint_jump schedule steps (cool faster: only the temperature and the
    // top-k anneal depend on the schedule position), never below cur_step = sprint_min.
    float    sprint_below = 0.0f;   // 0 = off
    int      sprint_jump  = 2;
    int      sprint_min   = 2;
    // "atk_sampler.pos_bias": acceptance ORDER key = entropy * (1 + pos_bias * i / L). Positive
    // commits the left of the canvas first (a mild left-to-right prior); the entropy BUDGET is
    // still summed from the raw entropies. 0 = the PR's pure entropy order.
    float    pos_bias = 0.0f;
    // "atk_sampler.accept_order": "entropy" (the PR) or "margin" -- order by the top-1/top-2
    // probability margin instead (host self-cond). Margin is often the better-calibrated
    // confidence for a discrete choice; the budget still sums entropy.
    std::string accept_order = "entropy";
    // "atk_sampler.bound_end": schedule the entropy budget from entropy_bound at the first step to
    // bound_end at the last (linear in schedule progress). < 0 = constant (the PR).
    float    bound_end = -1.0f;
    // "atk_sampler.temp_pos_gain": per-position temperature (host self-cond). Position i's top-k
    // distribution is re-tempered by T_i = T * (1 + g * z_i), z_i in [-1, 1] the entropy RANK of
    // the position (noisiest = +1): uncertain positions explore, confident ones commit. 0 = off.
    float    temp_pos_gain = 0.0f;
    // "atk_sampler.rep_*": spatial repetition penalty (host self-cond; plan item 6). Before
    // sampling position i, p_i(v) *= exp(-eta * sum_j w(|i-j|) * p_j(v)) over the +-rep_window
    // neighbours, w(d) = exp(-(d-1)/rep_decay), for tokens whose text has >= rep_min_chars
    // characters including a letter or digit (so punctuation, spaces and short pieces are never
    // penalised). The argmax canvas and the fed-back distribution take the penalty too.
    float    rep_penalty   = 0.0f;  // eta; 0 = off
    int      rep_window    = 8;
    float    rep_decay     = 4.0f;
    int      rep_min_chars = 3;
    // "atk_sampler.sc_sharpen": exponent on the fed-back distribution (host self-cond): p^gamma,
    // renormalised. > 1 sharpens (more commitment), < 1 softens (more exploration). 1 = the PR.
    float    sc_sharpen = 1.0f;
    // "atk_sampler.lateral_noise": a re-noised position draws its new token from a NEIGHBOUR's
    // top-k distribution with this probability (host self-cond) -- noise that is locally
    // plausible instead of uniform. Tried before warm noise. 0 = off.
    float    lateral_noise = 0.0f;
    // "atk_sampler.osc_*": oscillation hold (the cheap form of plan item 5, stall bifurcation).
    // A position whose argmax alternates A,B,A,B for osc_steps steps is oscillating; when 1 ..
    // osc_max_pos positions oscillate and the stop test has not passed, each is held for
    // osc_hold_steps steps at whichever of its two states had the lower entropy, then released.
    int      osc_steps      = 0;    // 0 = off
    int      osc_max_pos    = 4;
    int      osc_hold_steps = 3;
    // "atk_sampler.n_candidates": best-of-N. The request is run N times with seeds seed, seed+1,
    // ... (the prompt's KV is reused, so each costs its denoising steps only) and the canvas with
    // the lowest mean entropy is returned; every candidate's answer is in "atk_candidates".
    int      n_candidates = 1;
    // "atk_readout_sc": return each position's top-N self-conditioning entries (host self-cond)
    // in atk_canvas_out.self_cond, for the stateless verifier loop (see atk_canvas.self_cond).
    int      sc_out_k = 0;
    // "atk_canvas.self_cond": seed the FIRST step's self-conditioning from a previous readout
    // (host self-cond). This is what makes "denoise -> verify -> re-noise the failed spans ->
    // continue" a loop of stateless requests: the kept positions resume with their memory.
    bool                 has_sc_seed = false;
    int                  sc_seed_k   = 0;
    std::vector<int32_t> sc_seed_ids;    // sc_seed_k per position, row-major over the given positions
    std::vector<float>   sc_seed_probs;
    // "atk_canvas.soft_hold": a position that starts as its given token with its self-conditioning
    // seeded to that token at probability soft_hold_p (host self-cond) -- "prefer to keep this,
    // but you may fix it". A hard hold fixes the token; a soft hold only remembers it.
    std::vector<char>    soft_hold;
    float                soft_hold_p = 0.9f;
    // "atk_trajectory_text": each trajectory row also carries the argmax canvas as text.
    // "atk_stream_steps" (stream=true only): an "atk_step" event per denoising step with the
    // canvas text and entropy -- the denoising made visible.
    bool     trajectory_text = false;
    bool     stream_steps    = false;

    // features that read or edit the top-k distribution on the host
    bool needs_host_sc() const {
        return accept_order == "margin" || temp_pos_gain != 0.0f || rep_penalty > 0.0f ||
               sc_sharpen != 1.0f || lateral_noise > 0.0f || sc_out_k > 0 || has_sc_seed ||
               std::find(soft_hold.begin(), soft_hold.end(), (char) 1) != soft_hold.end();
    }
    // features that need the host sampling path (per-step host bookkeeping) on EVERY block
    bool needs_host_path() const {
        return needs_host_sc() || warm_start > 0.0f || warm_end > 0.0f || plateau_steps > 0 ||
               no_control || eog_tail || noise_text || noise_clear_sc || sprint_below > 0.0f ||
               pos_bias != 0.0f || bound_end >= 0.0f || osc_steps > 0 || stability_steps != 1 ||
               trajectory_text || stream_steps;
    }
};

struct diffusion_result {
    std::string answer;            // final response text (post channel-split / eog truncation)
    std::string full;             // full detokenized canvas (thought + response)
    int    prompt_tokens   = 0;    // prompt prefix tokens processed (encoder prefill)
    int    answer_tokens   = 0;    // tokens in the extracted answer
    int    n_blocks        = 0;    // canvas blocks run
    int    n_steps_total   = 0;    // denoising steps across all blocks
    int    canvas_tokens   = 0;    // n_blocks * canvas_length
    int    n_decode        = 0;    // total llama_decode() calls (prefill + denoise + commit)
    double prefill_ms      = 0.0;  // encoder-phase prompt prefill wall time
    double gen_ms          = 0.0;  // denoising block-loop wall time
    bool   ok              = false;
    std::string error;
    // ATK patch: the thought channel (text before the last "<channel|>"), whether an end-of-
    // generation token ended the answer (false = the canvases ran out: finish_reason "length"),
    // and whether the client went away mid-generation.
    std::string thought;
    bool   hit_eog         = false;
    bool   cancelled       = false;
    // ATK patch 2: prompt tokens whose KV was reused from the previous request
    int    cached_tokens   = 0;
    // ATK patch 2: the first canvas, read out (see diffusion_request::readout)
    bool                     has_readout = false;
    std::vector<llama_token> canvas0;
    std::vector<float>       canvas0_entropy;
    std::vector<int>         canvas0_flips;
    int                      canvas0_steps = 0;
    int                      canvas0_start = 0;
    // ATK patch 3: the first canvas's trajectory, one row per denoising step
    struct step_stat {
        int    step;          // 1-based, counting up
        int    cur_step;      // the schedule position (counts down; rises when a plateau adds steps)
        float  mean_entropy;  // over free (unheld) positions, before acceptance
        int    accepted;      // positions accepted this step
        int    flips;         // positions whose argmax changed since the previous step
        int    control;       // positions whose sample was a control token (not EOG)
        float  temp;
        double ms;
        bool   plateau;       // this step fired a plateau re-noise
        // ATK patch 4
        bool   sprint    = false; // this step skipped ahead on the schedule
        int    eog_pos   = -1;    // first position whose argmax is end-of-generation (-1: none)
        int    tail      = 0;     // positions excluded from the stop test as the post-EOG tail
        int    osc_holds = 0;     // positions under an oscillation hold after this step
        float  bound     = 0.0f;  // the entropy budget used this step
        std::string text;         // the argmax canvas as text (atk_trajectory_text)
    };
    std::vector<step_stat>   canvas0_trajectory;
    int                      canvas0_plateaus     = 0;
    bool                     canvas0_from_best    = false; // the returned canvas is the best seen, not the last
    float                    canvas0_best_entropy = 0.0f;
    float                    canvas0_last_entropy = 0.0f;
    int                      canvas0_control      = 0;     // control tokens (not EOG) in the returned canvas
    // ATK patch 4
    int                      canvas0_length    = 0;   // the canvas length this request ran with
    int                      canvas0_sprints   = 0;
    int                      canvas0_osc_holds = 0;   // oscillation-hold events
    int                      canvas0_eog_pos   = -1;  // first EOG position in the returned canvas
    int                      canvas0_tail_from = -1;  // first position the stop test stopped watching (-1: never)
    int                      canvas0_sc_cleared = 0;  // self-conditioning rows forgotten (plateau / noise_clear_sc)
    bool                     canvas0_host_sc   = false; // the request ran with host self-conditioning
    int                      sc_out_k = 0;            // self-conditioning readout, top-N per position (last step)
    std::vector<int32_t>     sc_out_ids;
    std::vector<float>       sc_out_probs;
    json                     candidates;              // best-of-N: one entry per candidate (empty otherwise)
};

// ATK patch: called from inside generate(). on_step runs before every denoising step and
// on_block after every finished canvas block (with that block's tokens); either returning
// false stops the generation (the client disconnected).
struct diffusion_hooks {
    std::function<bool(int block, int step, int n_steps)>               on_step;
    std::function<bool(int block, const std::vector<llama_token> & toks)> on_block;
    // ATK patch 4: after every denoising step on the host path, the argmax canvas and the
    // per-position entropy of that step (atk_stream_steps). Returning false stops the generation.
    std::function<bool(int block, int step, const std::vector<llama_token> & argmax,
                       const std::vector<float> & entropy, float mean_entropy, int accepted)> on_step_done;
};

// ------------------------------------------------------------------------------------------------
// ATK patch: the answer-extraction rule, shared by the final result and the streamed partial answer
// ------------------------------------------------------------------------------------------------
struct diffusion_answer {
    std::vector<llama_token> answer;    // after the last "<channel|>", up to the first eog token
    std::vector<llama_token> thought;   // before that "<channel|>" (empty when there is none)
    bool hit_eog    = false;            // an eog token ended the answer
    bool in_thought = false;            // a thought channel was opened and has not been closed yet
};

static llama_token diffusion_single_token(const llama_vocab * vocab, const char * text) {
    const std::vector<llama_token> t = common_tokenize(vocab, text, false, true);
    return t.size() == 1 ? t[0] : LLAMA_TOKEN_NULL;
}

static diffusion_answer diffusion_extract(const llama_vocab * vocab, const std::vector<llama_token> & generated,
                                          llama_token chan_open, llama_token chan_close) {
    diffusion_answer a;
    const int n_gen = (int) generated.size();
    int start = 0;
    if (chan_close != LLAMA_TOKEN_NULL) {
        for (int j = 0; j < n_gen; ++j) if (generated[j] == chan_close) start = j + 1;
    }
    if (start == 0 && chan_open != LLAMA_TOKEN_NULL) {
        for (int j = 0; j < n_gen && !llama_vocab_is_eog(vocab, generated[j]); ++j) {
            if (generated[j] == chan_open) { a.in_thought = true; break; }
        }
    }
    for (int j = start; j < n_gen; ++j) {
        if (llama_vocab_is_eog(vocab, generated[j])) { a.hit_eog = true; break; }
        a.answer.push_back(generated[j]);
    }
    if (start > 1) a.thought.assign(generated.begin(), generated.begin() + (start - 1));
    return a;
}

// ------------------------------------------------------------------------------------------------
// server metrics (llama-server analogue; protected by its own mutex)
// ------------------------------------------------------------------------------------------------
struct server_metrics {
    std::mutex mu;
    int64_t  t_start = 0;                 // process start (unix seconds)

    // cumulative counters
    uint64_t n_requests_total           = 0;
    uint64_t n_prompt_tokens_total      = 0;
    double   t_prompt_ms_total          = 0.0;
    uint64_t n_tokens_predicted_total   = 0;   // answer tokens
    double   t_predicted_ms_total       = 0.0; // denoise time
    uint64_t n_decode_total             = 0;
    uint64_t n_blocks_total             = 0;
    uint64_t n_steps_total              = 0;
    uint64_t n_canvas_tokens_total      = 0;

    // last-request gauges
    double   last_prompt_tps            = 0.0;
    double   last_predicted_tps         = 0.0;
    double   last_steps_per_second      = 0.0;
    double   last_canvas_tps            = 0.0;
    double   last_ms_per_step           = 0.0;

    std::atomic<int> n_processing{0};

    void add(const diffusion_result & r) {
        std::lock_guard<std::mutex> lk(mu);
        n_requests_total          += 1;
        n_prompt_tokens_total     += r.prompt_tokens;
        t_prompt_ms_total         += r.prefill_ms;
        n_tokens_predicted_total  += r.answer_tokens;
        t_predicted_ms_total      += r.gen_ms;
        n_decode_total            += r.n_decode;
        n_blocks_total            += r.n_blocks;
        n_steps_total             += r.n_steps_total;
        n_canvas_tokens_total     += r.canvas_tokens;
        last_prompt_tps      = r.prefill_ms > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
        last_predicted_tps   = r.gen_ms     > 0 ? r.answer_tokens * 1e3 / r.gen_ms     : 0.0;
        last_steps_per_second= r.gen_ms     > 0 ? r.n_steps_total * 1e3 / r.gen_ms     : 0.0;
        last_canvas_tps      = r.gen_ms     > 0 ? r.canvas_tokens * 1e3 / r.gen_ms     : 0.0;
        last_ms_per_step     = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total        : 0.0;
    }
};

// ------------------------------------------------------------------------------------------------
// server state (one model + one reused context == one slot, guarded by a mutex)
// ------------------------------------------------------------------------------------------------
struct diffusion_server {
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_memory_t      mem   = nullptr;
    llama_batch         batch{};
    common_chat_templates_ptr templates;

    int    canvas_length = DEF_CANVAS_LENGTH;
    int    n_steps       = DEF_MAX_DENOISE_STEPS;
    int    n_ctx         = 0;
    int    n_ub          = 0;       // ubatch == canvas_length (keeps the decoder gather graph small)
    int    n_vocab       = 0;
    int    topk_fixed    = 0;
    int    topk_start    = 0;
    int    topk_end      = 0;
    bool   topk_tail     = false;
    bool   use_gpu_sampling = false;
    bool   use_device_self_cond = false;
    bool   use_device_loop = false;
    int    device_early_stop_interval = 0;
    common_params_diffusion diffusion;
    std::string model_id;
    std::string model_path;
    std::string build_info = "llama.cpp diffusion-gemma-server (PR #24427) + ATK patch 4";

    server_metrics metrics;
    std::mutex gen_mutex;            // serialize generation (ctx is single-threaded == one slot)

    // ATK patch 2: what the KV cache holds, position by position (the last prompt plus the
    // canvases committed after it). Empty = nothing known, so the next request starts clean.
    std::vector<llama_token> kv_tokens;

    // ATK patch 4: the server's self-conditioning lives on the host (--no-diffusion-device-selfcond)
    bool host_self_cond = false;
    // ATK patch 4: the text-token table for noise_text (NORMAL / USER_DEFINED / BYTE attributes)
    std::vector<llama_token> noise_text_ids;
    // ATK patch 4: the pad token for eog_tail_pad (LLAMA_TOKEN_NULL when the vocab has none)
    llama_token pad_token = LLAMA_TOKEN_NULL;
    // ATK patch 4: lazily computed per token: may the repetition penalty touch it? (-1 = unknown)
    mutable std::vector<int8_t> rep_penalizable;

    void init_atk_tables() {
        noise_text_ids.clear();
        noise_text_ids.reserve(n_vocab);
        for (llama_token t = 0; t < n_vocab; ++t) {
            const int attr = (int) llama_vocab_get_attr(vocab, t);
            const bool text = (attr & (LLAMA_TOKEN_ATTR_NORMAL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_BYTE)) != 0;
            if (text && !llama_vocab_is_control(vocab, t) && !llama_vocab_is_eog(vocab, t)) noise_text_ids.push_back(t);
        }
        pad_token = llama_vocab_pad(vocab);
        rep_penalizable.assign(n_vocab, -1);
    }

    bool rep_may_penalize(llama_token t, int min_chars) const {
        if (t < 0 || t >= n_vocab) return false;
        if (rep_penalizable[t] >= 0) return rep_penalizable[t] == 1;
        const std::string piece = common_token_to_piece(vocab, t, false);
        int chars = 0; bool alnum = false;
        for (const unsigned char c : piece) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') continue;
            if ((c & 0xC0) != 0x80) ++chars;            // count UTF-8 lead bytes, not continuation bytes
            if (std::isalnum(c)) alnum = true;
            if (c >= 0x80) alnum = true;                 // non-ASCII text counts as letters
        }
        const bool ok = alnum && chars >= min_chars && !llama_vocab_is_control(vocab, t);
        rep_penalizable[t] = ok ? 1 : 0;
        return ok;
    }

    void invalidate_kv() {
        llama_memory_clear(mem, true);
        kv_tokens.clear();
    }

    std::string format_messages(const json & messages) const {
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.add_generation_prompt = true;
        // match the reference template default: emit the trailing thought-channel preamble
        // and skip the system turn (the inputs default of true renders the opposite)
        inputs.enable_thinking = false;
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }

    // causal prefill of `toks` starting at position pos0, chunked to n_ub (encoder phase).
    // increments *n_dec with the number of llama_decode() calls made.
    bool prefill_causal(const std::vector<llama_token> & toks, int pos0, int * n_dec) {
        llama_set_causal_attn(ctx, true);
        llama_set_diffusion_decoder_phase(ctx, false);
        llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);
        const int n = (int) toks.size();
        for (int off = 0; off < n; off += n_ub) {
            const int cnt = std::min(n_ub, n - off);
            batch.n_tokens = cnt;
            for (int i = 0; i < cnt; ++i) {
                batch.token[i]     = toks[off + i];
                batch.pos[i]       = pos0 + off + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (off + i == n - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) return false;
            ++(*n_dec);
        }
        return true;
    }

    // run the block-diffusion denoising loop for one request (mirrors diffusion-gemma-cli.cpp)
    diffusion_result generate(const std::vector<llama_token> & prompt_tokens, const diffusion_request & rq,
                              const diffusion_hooks * hooks = nullptr) {
        diffusion_result out;
        const int prefix_len = (int) prompt_tokens.size();
        int n_decode = 0;

        // ATK patch 4: the canvas length this request runs with (<= the server's, which is one ubatch)
        const int L = (rq.canvas_len_req > 0) ? rq.canvas_len_req : canvas_length;
        if (L < 8 || L > canvas_length) {
            out.error = "atk_canvas.length " + std::to_string(L) + " is outside 8.." + std::to_string(canvas_length) +
                        " (the server's canvas is the upper bound)";
            return out;
        }
        out.canvas0_length = L;
        // ATK patch 4: features that read or edit the top-k distribution need host self-conditioning
        if (rq.needs_host_sc() && !host_self_cond) {
            out.error = "this request uses a host self-conditioning feature (accept_order=margin, temp_pos_gain, rep_penalty, "
                        "sc_sharpen, lateral_noise, atk_readout_sc, atk_canvas.self_cond or soft_hold); launch the server with "
                        "--no-diffusion-device-selfcond to enable them (see /props host_self_cond)";
            return out;
        }
        if (rq.eog_tail_pad && pad_token == LLAMA_TOKEN_NULL) {
            out.error = "atk_sampler.eog_tail_pad: this vocabulary has no pad token";
            return out;
        }
        if (rq.noise_text && noise_text_ids.empty()) {
            out.error = "atk_sampler.noise_text: no text tokens found in the vocabulary";
            return out;
        }

        int max_canvases = rq.max_canvases;
        const int n_steps = std::max(rq.n_steps, 1);
        const int SC_K = diffusion.self_cond_top_k;
        const int fit = (n_ctx - prefix_len) / L - 1;
        if (fit < 1) {
            out.error = "prompt too long for context (n_ctx=" + std::to_string(n_ctx) + ")";
            return out;
        }
        if (max_canvases > fit) max_canvases = fit;
        if (max_canvases < 1)   max_canvases = 1;

        // ATK patch 2: PROMPT CACHE. The encoder is causal, so the K/V of a prompt prefix depends
        // on that prefix alone, and a request that starts with the same tokens as the last one
        // (a chat's history, Cognitive Coder's fixed system prompt and interfaces) need not
        // encode them again. The cache is full-size (the context default swa_full = true), so
        // dropping its tail is exact; if the memory refuses the partial removal, start clean.
        int reuse = 0;
        if (rq.cache_prompt && !kv_tokens.empty()) {
            const int lim = std::min((int) kv_tokens.size(), prefix_len);
            while (reuse < lim && kv_tokens[reuse] == prompt_tokens[reuse]) ++reuse;
        }
        if (reuse > 0 && !llama_memory_seq_rm(mem, 0, reuse, -1)) {
            reuse = 0;
        }
        if (reuse == 0) {
            llama_memory_clear(mem, true);
        }
        kv_tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + reuse);
        out.cached_tokens = reuse;

        std::mt19937 rng(rq.seed);
        std::uniform_int_distribution<int> rand_tok(0, n_vocab - 1);
        std::uniform_int_distribution<int> rand_text(0, std::max(1, (int) noise_text_ids.size()) - 1);
        std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);
        // ATK patch 4: one draw of noise -- uniform over the vocabulary (the PR) or over text tokens
        auto draw_noise = [&]() -> llama_token {
            return rq.noise_text ? noise_text_ids[rand_text(rng)] : rand_tok(rng);
        };

        // ---- ENCODER phase: prefill the prompt prefix (causal, no self-conditioning) ----
        const auto t_prefill0 = std::chrono::steady_clock::now();
        int n_past = reuse;
        if (prefix_len > reuse) {
            const std::vector<llama_token> rest(prompt_tokens.begin() + reuse, prompt_tokens.end());
            if (!prefill_causal(rest, reuse, &n_decode)) {
                invalidate_kv();
                out.error = "prompt prefill (encoder) decode failed";
                return out;
            }
            n_past = prefix_len;
        }
        kv_tokens = prompt_tokens;
        out.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill0).count();

        // ATK patch 2: the client's first canvas, checked before any denoising
        if (rq.has_canvas && ((int) rq.canvas_init.size() > L ||
                              rq.canvas_hold.size() != rq.canvas_init.size())) {
            out.error = "atk_canvas: " + std::to_string(rq.canvas_init.size()) + " tokens for a " +
                        std::to_string(L) + "-token canvas, or hold of a different length";
            return out;
        }
        for (size_t i = 0; rq.has_canvas && i < rq.canvas_init.size(); ++i) {
            if (rq.canvas_hold[i] && (rq.canvas_init[i] < 0 || rq.canvas_init[i] >= n_vocab)) {
                out.error = "atk_canvas: position " + std::to_string(i) + " is held but has no token";
                return out;
            }
        }
        // ATK patch 4: soft holds and the self-conditioning seed, checked the same way
        const bool any_soft = rq.has_canvas &&
            std::find(rq.soft_hold.begin(), rq.soft_hold.end(), (char) 1) != rq.soft_hold.end();
        if (any_soft && rq.soft_hold.size() != rq.canvas_init.size()) {
            out.error = "atk_canvas.soft_hold must have one entry per given token";
            return out;
        }
        for (size_t i = 0; any_soft && i < rq.soft_hold.size(); ++i) {
            if (rq.soft_hold[i] && (rq.canvas_init[i] < 0 || rq.canvas_init[i] >= n_vocab)) {
                out.error = "atk_canvas: position " + std::to_string(i) + " is soft-held but has no token";
                return out;
            }
            if (rq.soft_hold[i] && rq.canvas_hold[i]) {
                out.error = "atk_canvas: position " + std::to_string(i) + " is both held and soft-held";
                return out;
            }
        }
        if (rq.has_sc_seed) {
            if (rq.sc_seed_k < 1 || rq.sc_seed_k > SC_K) {
                out.error = "atk_canvas.self_cond.k must be 1.." + std::to_string(SC_K);
                return out;
            }
            if (rq.sc_seed_ids.size() != rq.sc_seed_probs.size() || rq.sc_seed_ids.size() % (size_t) rq.sc_seed_k != 0 ||
                rq.sc_seed_ids.size() / (size_t) rq.sc_seed_k > (size_t) L) {
                out.error = "atk_canvas.self_cond: ids/probs must be k entries per position, for at most " +
                            std::to_string(L) + " positions";
                return out;
            }
        }
        const bool want_readout = rq.readout || rq.has_canvas;
        // ATK patch 4: the host sampling path runs on EVERY block when any host feature is in use
        // (patch 3's features ran on block 0 only, because only the read-out block left the device loop)
        const bool host_path = want_readout || rq.needs_host_path();
        std::vector<int>   flips0;

        std::vector<llama_token> canvas(L);
        std::vector<llama_token> argmax_canvas(L, -1);
        std::vector<llama_token> prev_argmax(L, -1);
        std::vector<llama_token> accepted(L);
        int32_t stop_flag = 0;
        std::vector<int32_t>     sc_ids ((size_t) SC_K * L, 0);
        std::vector<float>       sc_probs((size_t) SC_K * L, 0.0f);
        std::vector<llama_token> generated;

        const bool pin_host_outputs = diffusion.pin_host_outputs;
        diffusion_cuda_host_pin pin_argmax_canvas(
                argmax_canvas.data(), argmax_canvas.size() * sizeof(argmax_canvas[0]), pin_host_outputs);
        diffusion_cuda_host_pin pin_stop_flag(&stop_flag, sizeof(stop_flag), pin_host_outputs);

        int n_blocks_run = 0, n_steps_total = 0;
        bool done = false;
        const int topk_max_requested =
            (rq.topk_start > 0 && rq.topk_end > 0) ? std::max(rq.topk_start, rq.topk_end) : rq.topk_fixed;
        const bool use_device_loop_request = use_device_loop && !host_path &&
            (topk_max_requested <= 0 || topk_max_requested <= GPU_SAMPLING_MAX_TOP_K);
        bool sc_host_valid = false;   // ATK patch 4: sc_ids/sc_probs hold the last step's distributions
        auto cancel = [&](const char * why) {
            llama_memory_seq_rm(mem, 0, n_past, -1);
            out.cancelled = true;
            out.error = why;
        };

        const auto t_gen0 = std::chrono::steady_clock::now();
        for (int block = 0; block < max_canvases && !done; ++block) {
            ++n_blocks_run;
            for (auto & t : canvas) t = draw_noise();
            std::fill(prev_argmax.begin(), prev_argmax.end(), -1);
            llama_set_diffusion_self_cond_topk(ctx, nullptr, nullptr, 0, 0);

            // ATK patch 2: lay out the client's first canvas, and where on the noise path to start
            const bool laid  = block == 0 && rq.has_canvas;
            const bool watch = block == 0 && want_readout;   // read-out accumulation (first canvas)
            int start_step = n_steps;
            if (laid) {
                for (size_t i = 0; i < rq.canvas_init.size(); ++i) {
                    const llama_token t = rq.canvas_init[i];
                    if (rq.canvas_hold[i]) {
                        canvas[i] = t;
                    } else {
                        const bool soft = any_soft && rq.soft_hold[i];
                        const float noise_i = (i < rq.canvas_noise_map.size()) ? rq.canvas_noise_map[i] : rq.canvas_noise;
                        if (soft) canvas[i] = t;   // ATK patch 4: a soft hold starts as its token, un-noised
                        else if (t >= 0 && t < n_vocab && rand_unif(rng) >= noise_i) canvas[i] = t;
                    }
                }
                float noise = std::min(std::max(rq.canvas_noise, 0.0f), 1.0f);
                if (rq.canvas_start >= 0.0f) {
                    noise = rq.canvas_start;
                } else if (!rq.canvas_noise_map.empty()) {
                    // ATK patch 3: the schedule starts at the mean noise of the given, unheld
                    // positions; positions past the given tokens are pure noise and count as 1.0.
                    double sum = 0.0; int n = 0;
                    for (size_t i = 0; i < rq.canvas_init.size(); ++i) {
                        if (rq.canvas_hold[i] || rq.canvas_init[i] < 0) continue;
                        sum += (i < rq.canvas_noise_map.size()) ? rq.canvas_noise_map[i] : rq.canvas_noise;
                        ++n;
                    }
                    const int n_blank = L - (int) rq.canvas_init.size();
                    if (n + n_blank > 0) noise = (float) ((sum + (double) n_blank) / (double) (n + n_blank));
                }
                noise = std::min(std::max(noise, 0.0f), 1.0f);
                start_step = std::max(1, std::min(n_steps, (int) lroundf(noise * (float) n_steps)));
            }
            // ATK patch 4: seed the first step's self-conditioning (host self-cond): the readout of a
            // previous request for the kept positions, and the soft holds' own tokens
            if (block == 0 && (rq.has_sc_seed || any_soft)) {
                std::fill(sc_ids.begin(),   sc_ids.end(),   0);
                std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);
                if (rq.has_sc_seed) {
                    const int n_given = (int) (rq.sc_seed_ids.size() / (size_t) rq.sc_seed_k);
                    for (int p = 0; p < n_given; ++p) {
                        for (int k = 0; k < rq.sc_seed_k; ++k) {
                            const int32_t id = rq.sc_seed_ids[(size_t) p * rq.sc_seed_k + k];
                            if (id < 0 || id >= n_vocab) continue;
                            sc_ids  [(size_t) p * SC_K + k] = id;
                            sc_probs[(size_t) p * SC_K + k] = std::min(std::max(rq.sc_seed_probs[(size_t) p * rq.sc_seed_k + k], 0.0f), 1.0f);
                        }
                    }
                }
                for (size_t i = 0; any_soft && i < rq.soft_hold.size(); ++i) {
                    if (!rq.soft_hold[i]) continue;
                    std::fill_n(sc_probs.begin() + i * SC_K, SC_K, 0.0f);
                    sc_ids  [i * SC_K] = rq.canvas_init[i];
                    sc_probs[i * SC_K] = rq.soft_hold_p;
                }
                llama_set_diffusion_self_cond_topk(ctx, sc_ids.data(), sc_probs.data(), SC_K, L);
                SRV_DBG("self-conditioning seeded: %s%s\n", rq.has_sc_seed ? "readout " : "", any_soft ? "soft-hold" : "");
            }
            if (watch) {
                flips0.assign(L, 0);
                out.canvas0_start = start_step;
            }
            // ATK patch 3: best-canvas retention and plateau re-noise, per canvas (ATK patch 4: on
            // every host-path block, not only the read-out one)
            float                    best_mean = INFINITY;
            std::vector<llama_token> best_argmax;
            std::vector<float>       best_entropy;
            std::vector<float>       last_entropy(L, 0.0f);
            float                    prev_mean = INFINITY;
            int                      stall     = 0;
            int                      plateaus  = 0;
            int                      step_no   = 0;
            // ATK patch 4: per-canvas state for the stop test, EOG tail, sprint and oscillation hold
            int                      stable_run   = 0;
            int                      eog_pos_prev = -1;
            int                      eog_run      = 0;
            int                      tail_from    = L;    // first position the stop test does not watch
            int                      tail_first   = -1;   // ... the first time it stopped watching one
            int                      sprints      = 0;
            int                      osc_events   = 0;
            int                      sc_cleared   = 0;
            std::vector<llama_token> prev2_argmax(L, -1);
            std::vector<float>       prev_entropy(L, 0.0f);
            std::vector<int>         alt_run(L, 0);
            std::vector<llama_token> osc_hold_tok(L, -1);
            std::vector<int>         osc_hold_left(L, 0);
            auto is_held = [&](int i) {
                return laid && i < (int) rq.canvas_hold.size() && rq.canvas_hold[i];
            };
            auto in_tail = [&](int i) { return i >= tail_from; };

            for (int cur_step = start_step; cur_step >= 1; --cur_step) {
                if (hooks && hooks->on_step && !hooks->on_step(block, start_step - cur_step + 1, start_step)) {
                    cancel("cancelled: the client went away");
                    return out;
                }
                ++n_steps_total;
                if (watch) ++out.canvas0_steps;
                ++step_no;
                const auto t_step0 = std::chrono::steady_clock::now();
                const float temp = rq.temp_min + (rq.temp_max - rq.temp_min) * ((float) cur_step / (float) n_steps);
                // schedule progress: 0 at start_step, 1 at the last step (warm noise, budget schedule)
                const float progress = (start_step > 1) ? (float) (start_step - cur_step) / (float) (start_step - 1) : 1.0f;
                // ATK patch 4: the entropy budget, scheduled when bound_end is given
                const float bound = (rq.bound_end >= 0.0f)
                    ? rq.entropy_bound + (rq.bound_end - rq.entropy_bound) * std::min(std::max(progress, 0.0f), 1.0f)
                    : rq.entropy_bound;

                int k_step = 0;
                if (rq.topk_start > 0 && rq.topk_end > 0) {
                    const float frac = (n_steps > 1) ? (float) (cur_step - 1) / (float) (n_steps - 1) : 0.0f;
                    k_step = (int) lroundf(rq.topk_end + (rq.topk_start - rq.topk_end) * frac);
                } else if (rq.topk_fixed > 0) {
                    k_step = rq.topk_fixed;
                }
                if (k_step <= 0 || k_step >= n_vocab) k_step = 0;
                const bool use_gpu_sampling_step = use_gpu_sampling &&
                    (k_step <= 0 || k_step <= GPU_SAMPLING_MAX_TOP_K);
                const bool use_device_self_cond_step = use_gpu_sampling_step && use_device_self_cond;
                // ATK patch 2: a laid-out or read-out canvas is updated on the host, where held
                // positions can be put back and per-position entropy is visible. Self-conditioning
                // stays wherever the server keeps it (on the card with the fused input), so the
                // only cost is copying the canvas's tokens and entropies per step.
                const bool use_device_loop_step = use_device_loop_request && use_gpu_sampling_step;

                llama_set_causal_attn(ctx, false);
                llama_set_diffusion_decoder_phase(ctx, true);
                llama_set_diffusion_gpu_sampling(ctx, use_gpu_sampling_step);
                batch.n_tokens = L;
                for (int j = 0; j < L; ++j) {
                    batch.token[j]     = canvas[j];
                    batch.pos[j]       = n_past + j;
                    batch.n_seq_id[j]  = 1;
                    batch.seq_id[j][0] = 0;
                    batch.logits[j]    = 1;
                }
                if (llama_decode(ctx, batch) != 0) {
                    invalidate_kv();
                    out.error = "llama_decode failed at denoising step " + std::to_string(cur_step);
                    return out;
                }
                ++n_decode;

                std::vector<float> entropy(L);
                std::vector<llama_token> sampled(L);

                if (use_device_loop_step) {
                    const int block_step = n_steps - cur_step + 1;
                    const bool use_device_early_stop = device_early_stop_interval > 0;
                    const bool reset_stop_state = use_device_early_stop && block_step == 1;
                    const bool check_stop = use_device_early_stop;
                    stop_flag = 0;
                    llama_diffusion_sample_params sample_params = {
                        /* .n_tokens              = */ L,
                        /* .top_k                 = */ k_step,
                        /* .self_cond_top_k       = */ SC_K,
                        /* .temperature           = */ temp,
                        /* .seed                  = */ rq.seed,
                        /* .step                  = */ (uint32_t) n_steps_total,
                        /* .top_k_tail_correction = */ rq.topk_tail,
                        /* .cuda_fast_top_k       = */ diffusion.cuda_fast_top_k,
                        /* .cuda_direct_self_cond = */ diffusion.cuda_direct_self_cond,
                        /* .cuda_final_tokens_on_stop = */ diffusion.cuda_final_tokens_on_stop && device_early_stop_interval > 0 && cur_step != 1,
                        /* .cuda_fused_top_k_sample = */ diffusion.cuda_fused_top_k_sample,
                        /* .cuda_parallel_full_softmax = */ diffusion.cuda_parallel_full_softmax,
                        /* .cuda_fused_full_softmax = */ diffusion.cuda_fused_full_softmax,
                    };
                    llama_diffusion_sample_result sample_result = {
                        /* .sampled         = */ nullptr,
                        /* .argmax          = */ nullptr,
                        /* .entropy         = */ nullptr,
                        /* .self_cond_ids   = */ nullptr,
                        /* .self_cond_probs = */ nullptr,
                        /* .final_tokens    = */ (cur_step == 1 || check_stop) ? argmax_canvas.data() : nullptr,
                        /* .stop            = */ check_stop ? &stop_flag : nullptr,
                        /* .entropy_bound   = */ rq.entropy_bound,
                        /* .confidence_threshold = */ rq.stop_entropy,
                        /* .stability_threshold  = */ STABILITY_THRESHOLD,
                        /* .update_canvas_on_device = */ cur_step > 1,
                        /* .update_stop_state_on_device = */ use_device_early_stop,
                        /* .check_stop_on_device = */ check_stop,
                        /* .reset_stop_state = */ reset_stop_state,
                    };
                    if (!llama_diffusion_sample_topk(ctx, &sample_params, &sample_result)) {
                        llama_memory_seq_rm(mem, 0, n_past, -1);
                        out.error = "CUDA diffusion device loop failed at denoising step " +
                            std::to_string(cur_step) + " (k=" + std::to_string(k_step) + ")";
                        return out;
                    }
                    llama_memory_seq_rm(mem, 0, n_past, -1);
                    if (check_stop && stop_flag != 0) {
                        break;
                    }
                    continue;
                }

                if (!use_device_self_cond_step) {
                    std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);
                    std::fill(sc_ids.begin(),   sc_ids.end(),   0);
                }

                if (use_gpu_sampling_step) {
                    llama_diffusion_sample_params sample_params = {
                        /* .n_tokens              = */ L,
                        /* .top_k                 = */ k_step,
                        /* .self_cond_top_k       = */ SC_K,
                        /* .temperature           = */ temp,
                        /* .seed                  = */ rq.seed,
                        /* .step                  = */ (uint32_t) n_steps_total,
                        /* .top_k_tail_correction = */ rq.topk_tail,
                        /* .cuda_fast_top_k       = */ diffusion.cuda_fast_top_k,
                        /* .cuda_direct_self_cond = */ diffusion.cuda_direct_self_cond,
                        /* .cuda_final_tokens_on_stop = */ diffusion.cuda_final_tokens_on_stop && device_early_stop_interval > 0 && cur_step != 1,
                        /* .cuda_fused_top_k_sample = */ diffusion.cuda_fused_top_k_sample,
                        /* .cuda_parallel_full_softmax = */ diffusion.cuda_parallel_full_softmax,
                        /* .cuda_fused_full_softmax = */ diffusion.cuda_fused_full_softmax,
                    };
                    llama_diffusion_sample_result sample_result = {
                        /* .sampled         = */ sampled.data(),
                        /* .argmax          = */ argmax_canvas.data(),
                        /* .entropy         = */ entropy.data(),
                        /* .self_cond_ids   = */ use_device_self_cond_step ? nullptr : sc_ids.data(),
                        /* .self_cond_probs = */ use_device_self_cond_step ? nullptr : sc_probs.data(),
                        /* .final_tokens    = */ nullptr,
                        /* .stop            = */ nullptr,
                        /* .entropy_bound   = */ 0.0f,
                        /* .confidence_threshold = */ 0.0f,
                        /* .stability_threshold  = */ 0,
                        /* .update_canvas_on_device = */ false,
                        /* .update_stop_state_on_device = */ false,
                        /* .check_stop_on_device = */ false,
                        /* .reset_stop_state = */ false,
                    };
                    if (!llama_diffusion_sample_topk(ctx, &sample_params, &sample_result)) {
                        llama_memory_seq_rm(mem, 0, n_past, -1);
                        out.error = "CUDA diffusion sampling failed at denoising step " +
                            std::to_string(cur_step) + " (k=" + std::to_string(k_step) + ")";
                        return out;
                    }
                } else if (k_step == 0) {
                    const float * logits = llama_get_logits(ctx);
                    const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
                    std::vector<float> probs(n_vocab);
                    std::vector<std::pair<float,int>> scheap; scheap.reserve(SC_K);
                    for (int j = 0; j < L; ++j) {
                        const float * lg = logits + (size_t) j * n_vocab;
                        float maxl = -INFINITY; int amax = 0;
                        scheap.clear();
                        for (int v = 0; v < n_vocab; ++v) {
                            const float x = lg[v] / temp;
                            if (x > maxl) { maxl = x; amax = v; }
                            if ((int) scheap.size() < SC_K) {
                                scheap.push_back({x, v});
                                std::push_heap(scheap.begin(), scheap.end(), cmp);
                            } else if (x > scheap.front().first) {
                                std::pop_heap(scheap.begin(), scheap.end(), cmp);
                                scheap.back() = {x, v};
                                std::push_heap(scheap.begin(), scheap.end(), cmp);
                            }
                        }
                        float sum = 0.0f;
                        for (int v = 0; v < n_vocab; ++v) { const float p = expf(lg[v] / temp - maxl); probs[v] = p; sum += p; }
                        float ent = 0.0f;
                        const float r = rand_unif(rng) * sum;
                        float cum = 0.0f; int tok = amax; bool picked = false;
                        for (int v = 0; v < n_vocab; ++v) {
                            const float p = probs[v] / sum;
                            if (p > 0.0f) ent -= p * logf(p);
                            cum += probs[v];
                            if (!picked && cum >= r) { tok = v; picked = true; }
                        }
                        int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                        float   * spr = sc_probs.data() + (size_t) j * SC_K;
                        int slot = 0;
                        for (auto & h : scheap) { sid[slot] = h.second; spr[slot] = expf(h.first - maxl) / sum; ++slot; }
                        entropy[j] = ent; sampled[j] = tok; argmax_canvas[j] = amax;
                    }
                } else {
                    const float * logits = llama_get_logits(ctx);
                    const auto cmp = [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; };
                    const int heap_k = std::max(k_step, SC_K);
                    std::vector<std::pair<float,int>> heap; heap.reserve(heap_k);
                    for (int j = 0; j < L; ++j) {
                        const float * lg = logits + (size_t) j * n_vocab;
                        float maxl = -INFINITY; int amax = 0;
                        heap.clear();
                        for (int v = 0; v < n_vocab; ++v) {
                            const float x = lg[v] / temp;
                            if (x > maxl) { maxl = x; amax = v; }
                            if ((int) heap.size() < heap_k) {
                                heap.push_back({x, v});
                                std::push_heap(heap.begin(), heap.end(), cmp);
                            } else if (x > heap.front().first) {
                                std::pop_heap(heap.begin(), heap.end(), cmp);
                                heap.back() = {x, v};
                                std::push_heap(heap.begin(), heap.end(), cmp);
                            }
                        }
                        std::sort(heap.begin(), heap.end(), [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });
                        float Zk = 0.0f;
                        for (int i = 0; i < k_step; ++i) { const float e = expf(heap[i].first - maxl); heap[i].first = e; Zk += e; }
                        float ent;
                        if (rq.topk_tail) {
                            double Zf = 0.0, T = 0.0;
                            for (int v = 0; v < n_vocab; ++v) {
                                const double d = (double) (lg[v] / temp) - (double) maxl;
                                const double e = exp(d); Zf += e; T += d * e;
                            }
                            ent = (float) (log(Zf) - T / Zf);
                        } else {
                            ent = 0.0f;
                            for (int i = 0; i < k_step; ++i) { const float q = heap[i].first / Zk; if (q > 0.0f) ent -= q * logf(q); }
                        }
                        const float r = rand_unif(rng) * Zk;
                        float cum = 0.0f; int tok = amax; bool picked = false;
                        for (int i = 0; i < k_step; ++i) { cum += heap[i].first; if (!picked && cum >= r) { tok = heap[i].second; picked = true; } }
                        int32_t * sid = sc_ids.data()   + (size_t) j * SC_K;
                        float   * spr = sc_probs.data() + (size_t) j * SC_K;
                        const int n_sc = std::min(k_step, SC_K);
                        for (int i = 0; i < n_sc; ++i) { sid[i] = heap[i].second; spr[i] = heap[i].first / Zk; }
                        entropy[j] = ent; sampled[j] = tok; argmax_canvas[j] = amax;
                    }
                }

                llama_memory_seq_rm(mem, 0, n_past, -1);
                const bool host_sc = !use_device_self_cond_step;   // the top-k rows are on the host this step
                sc_host_valid = host_sc;

                // ATK patch 4: edit the distributions on the host (host self-cond): the spatial
                // repetition penalty, per-position temperature, self-cond sharpening. Each row of
                // sc_probs is position j's top-k at this step's temperature (entries with p = 0 are
                // absent). The sample and the argmax are redrawn from the edited distribution; what is
                // fed back is the ORIGINAL distribution with the penalty applied, raised to sc_sharpen
                // (the per-position temperature changes how we sample, not what the model remembers).
                std::vector<float> margin;   // top-1 minus top-2 probability per position (accept_order = margin)
                const bool want_margin = host_sc && rq.accept_order == "margin";
                if (host_sc && (rq.rep_penalty > 0.0f || rq.temp_pos_gain != 0.0f || rq.sc_sharpen != 1.0f || want_margin)) {
                    std::vector<float> z(L, 0.0f);
                    if (rq.temp_pos_gain != 0.0f) {
                        std::vector<int> by_ent(L);
                        std::iota(by_ent.begin(), by_ent.end(), 0);
                        std::sort(by_ent.begin(), by_ent.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });
                        for (int r = 0; r < L; ++r) z[by_ent[r]] = (L > 1) ? (2.0f * (float) r / (float) (L - 1) - 1.0f) : 0.0f;
                    }
                    std::unordered_map<int32_t, std::vector<std::pair<int, float>>> where;   // token -> (position, p)
                    if (rq.rep_penalty > 0.0f) {
                        for (int j = 0; j < L; ++j) {
                            const int32_t * id = sc_ids.data()   + (size_t) j * SC_K;
                            const float   * pr = sc_probs.data() + (size_t) j * SC_K;
                            for (int k = 0; k < SC_K; ++k) {
                                if (pr[k] > 0.0f && rep_may_penalize(id[k], rq.rep_min_chars)) where[id[k]].push_back({j, pr[k]});
                            }
                        }
                    }
                    if (want_margin) margin.assign(L, 1.0f);
                    std::vector<float> pen(SC_K), samp(SC_K);
                    for (int j = 0; j < L; ++j) {
                        if (is_held(j)) continue;
                        int32_t * id = sc_ids.data()   + (size_t) j * SC_K;
                        float   * pr = sc_probs.data() + (size_t) j * SC_K;
                        float expo = 1.0f;
                        if (rq.temp_pos_gain != 0.0f) {
                            expo = 1.0f / std::max(1.0f + rq.temp_pos_gain * z[j], 0.05f);
                            expo = std::min(std::max(expo, 0.25f), 4.0f);
                        }
                        float Zs = 0.0f;
                        for (int k = 0; k < SC_K; ++k) {
                            pen[k] = 1.0f;
                            if (pr[k] <= 0.0f) { samp[k] = 0.0f; continue; }
                            if (rq.rep_penalty > 0.0f) {
                                const auto it = where.find(id[k]);
                                if (it != where.end()) {
                                    float s = 0.0f;
                                    for (const auto & pq : it->second) {
                                        const int d = std::abs(pq.first - j);
                                        if (d == 0 || d > rq.rep_window) continue;
                                        s += pq.second * expf(-(float) (d - 1) / std::max(rq.rep_decay, 1e-3f));
                                    }
                                    if (s > 0.0f) pen[k] = expf(-rq.rep_penalty * s);
                                }
                            }
                            samp[k] = (expo == 1.0f ? pr[k] : powf(pr[k], expo)) * pen[k];
                            Zs += samp[k];
                        }
                        if (Zs <= 0.0f) continue;
                        // redraw the sample and the argmax from the edited distribution
                        const float r = rand_unif(rng) * Zs;
                        float cum = 0.0f; int tok = -1, amax = 0; float best = -1.0f, second = -1.0f;
                        for (int k = 0; k < SC_K; ++k) {
                            if (samp[k] <= 0.0f) continue;
                            cum += samp[k];
                            if (tok < 0 && cum >= r) tok = id[k];
                            if (samp[k] > best) { second = best; best = samp[k]; amax = id[k]; }
                            else if (samp[k] > second) { second = samp[k]; }
                        }
                        if (tok < 0) tok = amax;
                        sampled[j] = tok; argmax_canvas[j] = amax;
                        if (want_margin) margin[j] = (best - std::max(second, 0.0f)) / Zs;
                        // what the model remembers: original x penalty, sharpened, renormalised
                        float Zf = 0.0f;
                        for (int k = 0; k < SC_K; ++k) {
                            if (pr[k] <= 0.0f) continue;
                            float f = pr[k] * pen[k];
                            if (rq.sc_sharpen != 1.0f) f = powf(f, rq.sc_sharpen);
                            pr[k] = f; Zf += f;
                        }
                        if (Zf > 0.0f) for (int k = 0; k < SC_K; ++k) pr[k] /= Zf;
                    }
                } else if (want_margin) {
                    margin.assign(L, 1.0f);
                    for (int j = 0; j < L; ++j) {
                        const float * pr = sc_probs.data() + (size_t) j * SC_K;
                        float best = -1.0f, second = -1.0f, Z = 0.0f;
                        for (int k = 0; k < SC_K; ++k) {
                            Z += pr[k];
                            if (pr[k] > best) { second = best; best = pr[k]; } else if (pr[k] > second) { second = pr[k]; }
                        }
                        if (Z > 0.0f) margin[j] = (best - std::max(second, 0.0f)) / Z;
                    }
                }

                // ATK patch 2: a held position is certain by definition: it IS the held token, it
                // spends none of the entropy budget, and it does not count toward stopping.
                for (int i = 0; i < L; ++i) {
                    if (is_held(i)) {
                        entropy[i] = 0.0f;
                        sampled[i] = argmax_canvas[i] = rq.canvas_init[i];
                    }
                }
                int flips_step = 0;
                if (watch) {
                    for (int i = 0; i < L; ++i) {
                        if (prev_argmax[i] >= 0 && prev_argmax[i] != argmax_canvas[i]) { ++flips0[i]; ++flips_step; }
                    }
                }

                // ATK patch 3: a control token (other than end-of-generation) sampled into a text
                // position is counted always, and refused when the request asks for it. This is
                // the diagnostic for the eaten-character defect: a control token renders as an
                // EMPTY piece, so the text on either side of it is glued together.
                std::vector<char> ctrl_sampled(L, 0);
                int n_ctrl = 0;
                for (int i = 0; i < L; ++i) {
                    if (is_held(i)) continue;
                    const llama_token t = sampled[i];
                    if (t >= 0 && t < n_vocab && llama_vocab_is_control(vocab, t) && !llama_vocab_is_eog(vocab, t)) {
                        ctrl_sampled[i] = 1; ++n_ctrl;
                    }
                }

                // ATK patch 4: the post-EOG tail. The first position whose argmax is end-of-generation,
                // and how many steps running it has been there; once stable, the positions after it
                // leave the stop test, the acceptance budget and the plateau's demolition set.
                int eog_pos = -1;
                for (int i = 0; i < L; ++i) {
                    const llama_token t = argmax_canvas[i];
                    if (t >= 0 && t < n_vocab && llama_vocab_is_eog(vocab, t)) { eog_pos = i; break; }
                }
                if (eog_pos >= 0 && eog_pos == eog_pos_prev) ++eog_run; else eog_run = (eog_pos >= 0) ? 1 : 0;
                eog_pos_prev = eog_pos;
                tail_from = (rq.eog_tail && eog_pos >= 0 && eog_run >= rq.eog_tail_stable) ? eog_pos + 1 : L;
                if (tail_from < L && tail_first < 0) tail_first = tail_from;

                // the stop test's view: unheld positions that are not the tail
                float ent_sum = 0.0f; int n_watch = 0;
                for (int i = 0; i < L; ++i) {
                    if (is_held(i) || in_tail(i)) continue;
                    ent_sum += entropy[i]; ++n_watch;
                }

                // acceptance order (ATK patch 4: pos_bias, or the top-1/top-2 margin), budget in raw entropy
                std::vector<float> key(L);
                for (int i = 0; i < L; ++i) {
                    key[i] = want_margin ? (1.0f - margin[i]) : entropy[i] * (1.0f + rq.pos_bias * (float) i / (float) std::max(L - 1, 1));
                }
                std::vector<int> order(L);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
                std::vector<char> accept_mask(L, 0);
                float prefix = 0.0f;
                for (int k = 0; k < L; ++k) {
                    if (prefix > bound) break;
                    const int i = order[k];
                    if (in_tail(i)) continue;                                   // ATK patch 4: the tail buys nothing
                    if (rq.no_control && ctrl_sampled[i]) continue;             // ATK patch 3
                    accept_mask[i] = 1; prefix += entropy[i];
                }
                for (int i = 0; i < L; ++i) if (accept_mask[i]) accepted[i] = sampled[i];

                const float mean_ent = ent_sum / (float) std::max(1, n_watch);
                // ATK patch 4: stability over the watched positions, for stability_steps running
                bool same = true;
                for (int i = 0; i < L && same; ++i) {
                    if (is_held(i) || in_tail(i)) continue;
                    if (prev_argmax[i] < 0 || prev_argmax[i] != argmax_canvas[i]) same = false;
                }
                stable_run = same ? stable_run + 1 : 0;
                const bool stable    = (STABILITY_THRESHOLD == 0) || stable_run >= std::max(rq.stability_steps, 1);
                const bool confident = mean_ent < rq.stop_entropy;

                // ATK patch 3: remember the best canvas seen (lowest mean entropy over watched positions)
                if (mean_ent < best_mean) {
                    best_mean    = mean_ent;
                    best_argmax  = argmax_canvas;
                    best_entropy = entropy;
                }

                // ATK patch 4: sprint -- already confident for where the schedule is: cool faster
                bool sprint_fired = false;
                if (!(stable && confident) && rq.sprint_below > 0.0f && mean_ent < rq.sprint_below && cur_step > rq.sprint_min) {
                    cur_step = std::max(rq.sprint_min, cur_step - std::max(rq.sprint_jump, 1));
                    ++sprints; sprint_fired = true;
                }

                // ATK patch 3: plateau re-noise. Runs only when the stop test has NOT passed.
                bool plateau_fired = false;
                std::vector<char> demolished;
                if (!(stable && confident) && rq.plateau_steps > 0 && plateaus < rq.plateau_max) {
                    if (prev_mean - mean_ent < rq.plateau_eps) ++stall; else stall = 0;
                    if (stall >= rq.plateau_steps) {
                        demolished.assign(L, 0);
                        const int n_demo = std::max(1, (int) lroundf(rq.plateau_frac * (float) std::max(1, n_watch)));
                        int c = 0;
                        // order[] is ascending in the acceptance key; walk it from the noisy end, by entropy
                        std::vector<int> by_ent(L);
                        std::iota(by_ent.begin(), by_ent.end(), 0);
                        std::sort(by_ent.begin(), by_ent.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });
                        for (int k = L - 1; k >= 0 && c < n_demo; --k) {
                            const int pidx = by_ent[k];
                            if (is_held(pidx) || in_tail(pidx)) continue;
                            demolished[pidx] = 1; ++c;
                            if (pidx > 0 && !is_held(pidx - 1)) demolished[pidx - 1] = 1;
                            if (pidx + 1 < L && !is_held(pidx + 1) && !in_tail(pidx + 1)) demolished[pidx + 1] = 1;
                        }
                        for (int i = 0; i < L; ++i) if (demolished[i]) accept_mask[i] = 0;
                        cur_step = std::min(start_step, cur_step + rq.plateau_extra);
                        stall = 0; ++plateaus; plateau_fired = true;
                        SRV_DBG("plateau %d/%d at step %d: %d positions demolished, +%d steps\n",
                                plateaus, rq.plateau_max, step_no, c, rq.plateau_extra);
                    }
                }
                prev_mean = mean_ent;

                // ATK patch 4: oscillation hold. alt_run counts steps of A,B,A,B alternation; when a
                // few positions oscillate and nothing else moves, pin each to its lower-entropy state.
                int osc_new = 0;
                if (rq.osc_steps > 0) {
                    for (int i = 0; i < L; ++i) {
                        if (is_held(i) || in_tail(i) || osc_hold_left[i] > 0) { alt_run[i] = 0; continue; }
                        const bool alternating = prev2_argmax[i] >= 0 && prev_argmax[i] >= 0 &&
                            argmax_canvas[i] == prev2_argmax[i] && argmax_canvas[i] != prev_argmax[i];
                        alt_run[i] = alternating ? alt_run[i] + 1 : 0;
                    }
                    if (!(stable && confident)) {
                        std::vector<int> osc;
                        for (int i = 0; i < L; ++i) if (alt_run[i] >= rq.osc_steps) osc.push_back(i);
                        if (!osc.empty() && (int) osc.size() <= rq.osc_max_pos) {
                            for (const int i : osc) {
                                osc_hold_tok[i]  = (entropy[i] <= prev_entropy[i]) ? argmax_canvas[i] : prev_argmax[i];
                                osc_hold_left[i] = std::max(rq.osc_hold_steps, 1);
                                alt_run[i] = 0; ++osc_new;
                            }
                            osc_events += osc_new;
                            SRV_DBG("oscillation hold at step %d: %d position(s) pinned for %d steps\n", step_no, osc_new, rq.osc_hold_steps);
                        }
                    }
                }
                int osc_active = 0;
                for (int i = 0; i < L; ++i) if (osc_hold_left[i] > 0) ++osc_active;

                int n_acc = 0;
                for (int i = 0; i < L; ++i) n_acc += accept_mask[i] ? 1 : 0;
                if (watch) {
                    diffusion_result::step_stat st;
                    st.step = step_no; st.cur_step = cur_step; st.mean_entropy = mean_ent; st.accepted = n_acc;
                    st.flips = flips_step; st.control = n_ctrl; st.temp = temp; st.plateau = plateau_fired;
                    st.sprint = sprint_fired; st.eog_pos = eog_pos; st.tail = L - tail_from; st.osc_holds = osc_active; st.bound = bound;
                    st.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step0).count();
                    if (rq.trajectory_text) st.text = common_detokenize(vocab, argmax_canvas, false);
                    out.canvas0_trajectory.push_back(st);
                }
                last_entropy = entropy;
                if (hooks && hooks->on_step_done && rq.stream_steps &&
                    !hooks->on_step_done(block, step_no, argmax_canvas, entropy, mean_ent, n_acc)) {
                    cancel("cancelled: the client went away");
                    return out;
                }

                if (stable && confident) break;
                prev2_argmax = prev_argmax;
                prev_argmax  = argmax_canvas;
                prev_entropy = entropy;

                if (host_sc) {
                    llama_set_diffusion_self_cond_topk(ctx, sc_ids.data(), sc_probs.data(), SC_K, L);
                }

                // ATK patch 3: warm noise. A rejected position keeps the model's own sample with
                // probability rho instead of a uniform random token (never a control token, and
                // never for a position a plateau just demolished -- those get pure noise, which
                // is the point of demolishing them). rho runs warm_start -> warm_end over the
                // schedule. ATK patch 4 adds, in this order: an oscillation hold (the position is
                // pinned), the pad token for the post-EOG tail, lateral noise (a neighbour's
                // distribution), then warm noise, then uniform noise -- over text tokens only with
                // noise_text -- and the forgetting of self-conditioning for demolished / re-noised
                // positions.
                const float rho = std::min(std::max(rq.warm_start + (rq.warm_end - rq.warm_start) * progress, 0.0f), 1.0f);
                std::vector<int32_t> clear_pos;
                for (int i = 0; i < L; ++i) {
                    if (osc_hold_left[i] > 0) { canvas[i] = osc_hold_tok[i]; --osc_hold_left[i]; continue; }
                    if (accept_mask[i]) { canvas[i] = accepted[i]; continue; }
                    if (in_tail(i) && rq.eog_tail_pad) { canvas[i] = pad_token; continue; }
                    const bool demo = !demolished.empty() && demolished[i];
                    bool drew = false;
                    if (!demo && host_sc && rq.lateral_noise > 0.0f && rand_unif(rng) < rq.lateral_noise) {
                        int j = (rand_unif(rng) < 0.5f) ? i - 1 : i + 1;
                        if (j < 0 || j >= L || is_held(j)) j = (j < 0) ? i + 1 : i - 1;
                        if (j >= 0 && j < L && !is_held(j)) {
                            const int32_t * id = sc_ids.data()   + (size_t) j * SC_K;
                            const float   * pr = sc_probs.data() + (size_t) j * SC_K;
                            float Z = 0.0f;
                            for (int k = 0; k < SC_K; ++k) Z += pr[k];
                            if (Z > 0.0f) {
                                const float r = rand_unif(rng) * Z;
                                float cum = 0.0f;
                                for (int k = 0; k < SC_K; ++k) {
                                    cum += pr[k];
                                    if (pr[k] > 0.0f && cum >= r) { canvas[i] = id[k]; drew = true; break; }
                                }
                            }
                        }
                    }
                    if (!drew) {
                        if (rho > 0.0f && !demo && !ctrl_sampled[i] && rand_unif(rng) < rho) canvas[i] = sampled[i];
                        else canvas[i] = draw_noise();
                    }
                    if (demo ? rq.demolish_clear_sc : rq.noise_clear_sc) clear_pos.push_back(i);
                }
                for (int i = 0; laid && i < (int) rq.canvas_hold.size(); ++i) {
                    if (rq.canvas_hold[i]) canvas[i] = rq.canvas_init[i];
                }
                if (!clear_pos.empty()) {
                    const int32_t n = llama_diffusion_self_cond_clear(ctx, clear_pos.data(), (int32_t) clear_pos.size());
                    if (n > 0) sc_cleared += n;
                    else if (n < 0) SRV_DBG("self-cond clear: no live state at step %d (%zu positions)\n", step_no, clear_pos.size());
                }
            }
            // ATK patch 3: the last step's mean entropy, and the swap to the best canvas when the last
            // is worse. The swap is made in argmax_canvas ITSELF (element copy, not reassignment: the
            // vector is pinned host memory) so that the readout, the text in choices and the canvas
            // committed to the KV all agree. ATK patch 4: on every host-path block.
            if (host_path) {
                float last_sum = 0.0f; int n_last = 0;
                for (int i = 0; i < L; ++i) {
                    if (is_held(i) || in_tail(i)) continue;
                    last_sum += last_entropy[i]; ++n_last;
                }
                const float last_mean = last_sum / (float) std::max(1, n_last);
                bool from_best = false;
                if (rq.keep_best && !best_argmax.empty() && best_mean < last_mean - 1e-6f) {
                    std::copy(best_argmax.begin(), best_argmax.end(), argmax_canvas.begin());
                    last_entropy = best_entropy;
                    from_best = true;
                }
                if (watch) {
                    out.canvas0_last_entropy = last_mean;
                    out.canvas0_best_entropy = (best_mean < INFINITY) ? best_mean : last_mean;
                    out.canvas0_plateaus     = plateaus;
                    out.canvas0_from_best    = from_best;
                    out.canvas0_sprints      = sprints;
                    out.canvas0_osc_holds    = osc_events;
                    out.canvas0_sc_cleared   = sc_cleared;
                    out.canvas0_tail_from    = tail_first;
                    out.canvas0_host_sc      = sc_host_valid;
                    out.has_readout     = true;
                    out.canvas0         = argmax_canvas;
                    out.canvas0_entropy = last_entropy;
                    out.canvas0_flips   = flips0;
                    int n_ctrl_out = 0, eog_out = -1;
                    for (int i = 0; i < L; ++i) {
                        const llama_token t = argmax_canvas[i];
                        if (t < 0 || t >= n_vocab) continue;
                        if (llama_vocab_is_eog(vocab, t)) { if (eog_out < 0) eog_out = i; }
                        else if (llama_vocab_is_control(vocab, t)) ++n_ctrl_out;
                    }
                    out.canvas0_control = n_ctrl_out;
                    out.canvas0_eog_pos = eog_out;
                    // ATK patch 4: the self-conditioning readout -- the LAST step's top-N per position
                    // (when the returned canvas is the best seen rather than the last, the memory is
                    // still the last step's; the client re-noises what it rejects anyway)
                    if (rq.sc_out_k > 0 && sc_host_valid) {
                        const int N = std::min(rq.sc_out_k, SC_K);
                        out.sc_out_k = N;
                        out.sc_out_ids.assign((size_t) N * L, -1);
                        out.sc_out_probs.assign((size_t) N * L, 0.0f);
                        std::vector<std::pair<float, int32_t>> row;
                        for (int j = 0; j < L; ++j) {
                            row.clear();
                            for (int k = 0; k < SC_K; ++k) {
                                const float p = sc_probs[(size_t) j * SC_K + k];
                                if (p > 0.0f) row.push_back({p, sc_ids[(size_t) j * SC_K + k]});
                            }
                            const int n = std::min(N, (int) row.size());
                            std::partial_sort(row.begin(), row.begin() + n, row.end(),
                                              [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) { return a.first > b.first; });
                            for (int k = 0; k < n; ++k) {
                                out.sc_out_ids  [(size_t) j * N + k] = row[k].second;
                                out.sc_out_probs[(size_t) j * N + k] = row[k].first;
                            }
                        }
                    }
                }
                SRV_DBG("block %d: %d steps, last mean entropy %.4f, best %.4f%s, plateaus %d, sprints %d, osc holds %d, sc cleared %d, tail from %d\n",
                        block, step_no, last_mean, best_mean, from_best ? " (returned)" : "", plateaus, sprints, osc_events, sc_cleared, tail_first);
            }

            const std::vector<llama_token> & block_out = argmax_canvas;
            generated.insert(generated.end(), block_out.begin(), block_out.end());
            if (!rq.ignore_eos) {
                for (int j = 0; j < L; ++j) {
                    if (llama_vocab_is_eog(vocab, block_out[j])) { done = true; break; }
                }
            }
            if (hooks && hooks->on_block && !hooks->on_block(block, block_out)) {
                out.cancelled = true;
                out.error = "cancelled: the client went away";
                return out;
            }
            if (!done && block + 1 < max_canvases) {
                if (!prefill_causal(block_out, n_past, &n_decode)) {
                    invalidate_kv();
                    out.error = "canvas commit (encoder) decode failed";
                    return out;
                }
                n_past += L;
                kv_tokens.insert(kv_tokens.end(), block_out.begin(), block_out.end());
            }
        }
        out.gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen0).count();
        out.full   = common_detokenize(vocab, generated, false);

        // extract the final response: after the last "<channel|>" close, until the first eog token
        // (ATK patch: the rule lives in diffusion_extract so the streamed partial answer uses it too)
        const diffusion_answer ex = diffusion_extract(vocab, generated,
            diffusion_single_token(vocab, "<|channel>"), diffusion_single_token(vocab, "<channel|>"));
        const std::vector<llama_token> & answer = ex.answer;
        out.hit_eog = ex.hit_eog;
        if (!ex.thought.empty()) out.thought = common_detokenize(vocab, ex.thought, false);
        std::string ans = common_detokenize(vocab, answer, false);
        {
            std::string s = ans;
            size_t h = s.find_first_not_of(" \n\t"); if (h != std::string::npos) s = s.substr(h);
            const size_t half = s.size() / 2;
            if (half > 0 && s.compare(0, half, s, s.size() - half, half) == 0) ans = s.substr(0, half);
        }

        out.answer        = ans;
        out.prompt_tokens = prefix_len;
        out.answer_tokens = (int) answer.size();
        out.n_blocks      = n_blocks_run;
        out.n_steps_total = n_steps_total;
        out.canvas_tokens = n_blocks_run * L;
        out.n_decode      = n_decode;
        out.ok            = true;
        return out;
    }
};

// ------------------------------------------------------------------------------------------------
// JSON / log helpers
// ------------------------------------------------------------------------------------------------
static std::atomic<uint64_t> g_req_counter{0};
static std::string gen_id(const char * prefix) { return std::string(prefix) + "-" + std::to_string(g_req_counter.fetch_add(1)); }
static json error_json(const std::string & msg, const std::string & type, int code) {
    return json{{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
}

// llama-server `timings` object, with diffusion fields. prompt_* describes the encoder prefill;
// predicted_* describes the answer tokens; the nested `diffusion` object describes the denoising.
static json timings_json(const diffusion_result & r) {
    const double ppt = r.prompt_tokens > 0 ? r.prefill_ms / r.prompt_tokens : 0.0;
    const double pps = r.prefill_ms    > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
    const double dpt = r.answer_tokens > 0 ? r.gen_ms / r.answer_tokens : 0.0;
    const double dps = r.gen_ms        > 0 ? r.answer_tokens * 1e3 / r.gen_ms : 0.0;
    const double sps = r.gen_ms        > 0 ? r.n_steps_total * 1e3 / r.gen_ms : 0.0;
    const double cps = r.gen_ms        > 0 ? r.canvas_tokens * 1e3 / r.gen_ms : 0.0;
    const double mps = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total : 0.0;
    return json{
        {"cache_n", r.cached_tokens},              // ATK patch 2: prompt tokens reused from the cache
        {"prompt_n", r.prompt_tokens},
        {"prompt_ms", r.prefill_ms},
        {"prompt_per_token_ms", ppt},
        {"prompt_per_second", pps},
        {"predicted_n", r.answer_tokens},
        {"predicted_ms", r.gen_ms},
        {"predicted_per_token_ms", dpt},
        {"predicted_per_second", dps},
        {"diffusion", {
            {"n_blocks", r.n_blocks},
            {"n_steps", r.n_steps_total},
            {"canvas_tokens", r.canvas_tokens},
            {"ms_per_step", mps},
            {"steps_per_second", sps},
            {"canvas_tokens_per_second", cps},
            {"n_decode", r.n_decode},
        }},
    };
}

static json usage_json(const diffusion_result & r) {
    const int reasoning = r.canvas_tokens - r.answer_tokens;
    return json{
        {"prompt_tokens", r.prompt_tokens},
        {"completion_tokens", r.canvas_tokens},
        {"total_tokens", r.prompt_tokens + r.canvas_tokens},
        {"prompt_tokens_details", {{"cached_tokens", r.cached_tokens}}},
        {"completion_tokens_details", {{"reasoning_tokens", reasoning > 0 ? reasoning : 0}}},
    };
}

// per-request timing log, llama_perf-style but with the denoise phase substituted for AR eval
static void log_timings(const diffusion_result & r) {
    const double ppt = r.prompt_tokens > 0 ? r.prefill_ms / r.prompt_tokens : 0.0;
    const double pps = r.prefill_ms    > 0 ? r.prompt_tokens * 1e3 / r.prefill_ms : 0.0;
    const double mps = r.n_steps_total > 0 ? r.gen_ms / r.n_steps_total : 0.0;
    const double sps = r.gen_ms        > 0 ? r.n_steps_total * 1e3 / r.gen_ms : 0.0;
    const double cps = r.gen_ms        > 0 ? r.canvas_tokens * 1e3 / r.gen_ms : 0.0;
    SLT_INF(0, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second), %d reused\n",
            r.prefill_ms, r.prompt_tokens, ppt, pps, r.cached_tokens);
    SLT_INF(0, "    denoise time = %10.2f ms / %5d steps  (%8.2f ms per step,  %8.2f steps per second)\n",
            r.gen_ms, r.n_steps_total, mps, sps);
    SLT_INF(0, "      gen tokens = %5d answer | %5d canvas over %d block(s) (%8.2f canvas tok/s)\n",
            r.answer_tokens, r.canvas_tokens, r.n_blocks, cps);
    SLT_INF(0, "      total time = %10.2f ms / %5d decode call(s)\n",
            r.prefill_ms + r.gen_ms, r.n_decode);
}

static diffusion_request request_from_body(const json & body, const diffusion_server & srv, uint32_t default_seed) {
    diffusion_request rq;
    rq.canvas_length = srv.canvas_length;
    rq.n_steps       = srv.n_steps;
    rq.seed          = default_seed;
    rq.topk_fixed    = srv.topk_fixed;
    rq.topk_start    = srv.topk_start;
    rq.topk_end      = srv.topk_end;
    rq.topk_tail     = srv.topk_tail;
    int max_tokens = 0;
    if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())                       max_tokens = body["max_tokens"].get<int>();
    else if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer()) max_tokens = body["max_completion_tokens"].get<int>();
    rq.max_canvases = max_tokens > 0 ? (max_tokens + rq.canvas_length - 1) / rq.canvas_length : 1;
    if (body.contains("diffusion_steps") && body["diffusion_steps"].is_number_integer()) {
        rq.n_steps = std::max(body["diffusion_steps"].get<int>(), 1);
    } else if (body.contains("n_denoise_steps") && body["n_denoise_steps"].is_number_integer()) {
        rq.n_steps = std::max(body["n_denoise_steps"].get<int>(), 1);
    }
    if (body.contains("top_k") && body["top_k"].is_number_integer()) {
        rq.topk_fixed = body["top_k"].get<int>();
        rq.topk_start = 0;
        rq.topk_end   = 0;
    }
    if (body.contains("top_k_start") && body["top_k_start"].is_number_integer()) rq.topk_start = body["top_k_start"].get<int>();
    if (body.contains("top_k_end")   && body["top_k_end"].is_number_integer())   rq.topk_end   = body["top_k_end"].get<int>();
    if (body.contains("top_k_tail_correction") && body["top_k_tail_correction"].is_boolean()) {
        rq.topk_tail = body["top_k_tail_correction"].get<bool>();
    }
    const int forced_top_k = std::max(srv.diffusion.force_top_k, 0);
    if (forced_top_k > 0) {
        rq.topk_fixed = forced_top_k;
        rq.topk_start = 0;
        rq.topk_end   = 0;
    }
    if (body.contains("seed")  && body["seed"].is_number_integer()) rq.seed = (uint32_t) body["seed"].get<int64_t>();
    if (body.contains("ignore_eos") && body["ignore_eos"].is_boolean()) rq.ignore_eos = body["ignore_eos"].get<bool>();

    // ---- ATK patch 2 -------------------------------------------------------------------------
    if (body.contains("atk_sampler") && body["atk_sampler"].is_object()) {
        const json & sm = body["atk_sampler"];
        auto num = [&](const char * k, float & dst, float lo, float hi) {
            if (sm.contains(k) && sm[k].is_number()) dst = std::min(std::max(sm[k].get<float>(), lo), hi);
        };
        num("entropy_bound", rq.entropy_bound, 0.0f, 1000.0f);
        num("stop_entropy",  rq.stop_entropy,  0.0f, 1000.0f);
        num("temp_max",      rq.temp_max,      0.01f, 10.0f);
        num("temp_min",      rq.temp_min,      0.01f, 10.0f);
        // ATK patch 3
        num("warm_start",    rq.warm_start,    0.0f, 1.0f);
        num("warm_end",      rq.warm_end,      0.0f, 1.0f);
        num("plateau_eps",   rq.plateau_eps,   0.0f, 10.0f);
        num("plateau_frac",  rq.plateau_frac,  0.0f, 1.0f);
        auto integer = [&](const char * k, int & dst, int lo, int hi) {
            if (sm.contains(k) && sm[k].is_number_integer()) dst = std::min(std::max((int) sm[k].get<int64_t>(), lo), hi);
        };
        integer("plateau_steps", rq.plateau_steps, 0, 1000);
        integer("plateau_extra", rq.plateau_extra, 0, 1000);
        integer("plateau_max",   rq.plateau_max,   0, 100);
        auto boolean = [&](const char * k, bool & dst) {
            if (sm.contains(k) && sm[k].is_boolean()) dst = sm[k].get<bool>();
        };
        boolean("keep_best",  rq.keep_best);
        boolean("no_control", rq.no_control);
        // ATK patch 4
        integer("canvas_length",   rq.canvas_len_req,  0, 4096);
        integer("stability_steps", rq.stability_steps, 1, 100);
        boolean("eog_tail",        rq.eog_tail);
        integer("eog_tail_stable", rq.eog_tail_stable, 1, 100);
        boolean("eog_tail_pad",    rq.eog_tail_pad);
        boolean("noise_text",      rq.noise_text);
        boolean("noise_clear_sc",  rq.noise_clear_sc);
        boolean("demolish_clear_sc", rq.demolish_clear_sc);
        num("sprint_below",      rq.sprint_below,  0.0f, 1000.0f);
        integer("sprint_jump",   rq.sprint_jump,   1, 1000);
        integer("sprint_min",    rq.sprint_min,    1, 1000);
        num("pos_bias",          rq.pos_bias,     -0.99f, 100.0f);
        if (sm.contains("accept_order") && sm["accept_order"].is_string()) {
            const std::string v = sm["accept_order"].get<std::string>();
            if (v == "entropy" || v == "margin") rq.accept_order = v;
        }
        num("bound_end",         rq.bound_end,    -1.0f, 1000.0f);
        num("temp_pos_gain",     rq.temp_pos_gain, -0.9f, 10.0f);
        num("rep_penalty",       rq.rep_penalty,   0.0f, 100.0f);
        integer("rep_window",    rq.rep_window,    1, 4096);
        num("rep_decay",         rq.rep_decay,     0.001f, 1000.0f);
        integer("rep_min_chars", rq.rep_min_chars, 1, 64);
        num("sc_sharpen",        rq.sc_sharpen,    0.05f, 20.0f);
        num("lateral_noise",     rq.lateral_noise, 0.0f, 1.0f);
        integer("osc_steps",     rq.osc_steps,     0, 1000);
        integer("osc_max_pos",   rq.osc_max_pos,   1, 4096);
        integer("osc_hold_steps", rq.osc_hold_steps, 1, 1000);
        integer("n_candidates",  rq.n_candidates,  1, 64);
    }
    if (body.contains("atk_canvas") && body["atk_canvas"].is_object()) {
        const json & cv = body["atk_canvas"];
        rq.has_canvas = true;
        if (cv.contains("tokens") && cv["tokens"].is_array()) {
            for (const auto & t : cv["tokens"]) rq.canvas_init.push_back(t.is_number_integer() ? (llama_token) t.get<int64_t>() : -1);
        }
        if (cv.contains("hold") && cv["hold"].is_array()) {
            for (const auto & h : cv["hold"]) rq.canvas_hold.push_back((char) ((h.is_boolean() && h.get<bool>()) || (h.is_number_integer() && h.get<int64_t>() != 0)));
        } else {
            rq.canvas_hold.assign(rq.canvas_init.size(), 0);
        }
        if (cv.contains("noise") && cv["noise"].is_number()) rq.canvas_noise = std::min(std::max(cv["noise"].get<float>(), 0.0f), 1.0f);
        // ATK patch 3
        if (cv.contains("noise_map") && cv["noise_map"].is_array()) {
            for (const auto & v : cv["noise_map"]) {
                rq.canvas_noise_map.push_back(v.is_number() ? std::min(std::max(v.get<float>(), 0.0f), 1.0f) : rq.canvas_noise);
            }
        }
        if (cv.contains("start") && cv["start"].is_number()) rq.canvas_start = std::min(std::max(cv["start"].get<float>(), 0.0f), 1.0f);
        // ATK patch 4
        if (cv.contains("length") && cv["length"].is_number_integer()) rq.canvas_len_req = (int) cv["length"].get<int64_t>();
        if (cv.contains("soft_hold") && cv["soft_hold"].is_array()) {
            for (const auto & h : cv["soft_hold"]) rq.soft_hold.push_back((char) ((h.is_boolean() && h.get<bool>()) || (h.is_number_integer() && h.get<int64_t>() != 0)));
        }
        if (cv.contains("soft_hold_p") && cv["soft_hold_p"].is_number()) rq.soft_hold_p = std::min(std::max(cv["soft_hold_p"].get<float>(), 0.0f), 1.0f);
        if (cv.contains("self_cond") && cv["self_cond"].is_object()) {
            const json & sc = cv["self_cond"];
            const int k = (sc.contains("k") && sc["k"].is_number_integer()) ? (int) sc["k"].get<int64_t>() : 0;
            if (k > 0 && sc.contains("ids") && sc["ids"].is_array() && sc.contains("probs") && sc["probs"].is_array() &&
                sc["ids"].size() == sc["probs"].size()) {
                rq.has_sc_seed = true;
                rq.sc_seed_k   = k;
                for (size_t p = 0; p < sc["ids"].size(); ++p) {
                    const json & row_i = sc["ids"][p];
                    const json & row_p = sc["probs"][p];
                    for (int j = 0; j < k; ++j) {
                        const bool ok = row_i.is_array() && row_p.is_array() && j < (int) row_i.size() && j < (int) row_p.size() &&
                                        row_i[j].is_number_integer() && row_p[j].is_number();
                        rq.sc_seed_ids.push_back(ok ? (int32_t) row_i[j].get<int64_t>() : -1);
                        rq.sc_seed_probs.push_back(ok ? row_p[j].get<float>() : 0.0f);
                    }
                }
            }
        }
    }
    if (body.contains("atk_readout") && body["atk_readout"].is_boolean()) rq.readout = body["atk_readout"].get<bool>();
    if (body.contains("atk_cache_prompt") && body["atk_cache_prompt"].is_boolean()) rq.cache_prompt = body["atk_cache_prompt"].get<bool>();
    // ATK patch 4
    if (body.contains("atk_readout_sc") && body["atk_readout_sc"].is_number_integer()) rq.sc_out_k = std::min(std::max((int) body["atk_readout_sc"].get<int64_t>(), 0), 4096);
    if (body.contains("atk_trajectory_text") && body["atk_trajectory_text"].is_boolean()) rq.trajectory_text = body["atk_trajectory_text"].get<bool>();
    if (body.contains("atk_stream_steps") && body["atk_stream_steps"].is_boolean()) rq.stream_steps = body["atk_stream_steps"].get<bool>();
    if (rq.sc_out_k > 0 || rq.n_candidates > 1) rq.readout = true;   // both need the per-step bookkeeping
    // a shorter canvas: max_tokens buys more (shorter) blocks
    if (rq.canvas_len_req > 0 && max_tokens > 0) {
        rq.max_canvases = (max_tokens + rq.canvas_len_req - 1) / rq.canvas_len_req;
    }
    return rq;
}

// ATK patch 2: the first canvas as the client asked to see it (atk_canvas / atk_readout)
// Each token's text goes out as HEX: a byte-fallback token is part of a UTF-8 character, which
// is not a valid JSON string on its own; the client joins the bytes and decodes the whole.
static json readout_json(const diffusion_server & srv, const diffusion_result & r) {
    static const char * HEX = "0123456789abcdef";
    json pieces = json::array(), eog = json::array();
    for (const llama_token t : r.canvas0) {
        const std::string raw = common_token_to_piece(srv.vocab, t, false);
        std::string hex;
        hex.reserve(raw.size() * 2);
        for (const unsigned char c : raw) { hex.push_back(HEX[c >> 4]); hex.push_back(HEX[c & 15]); }
        pieces.push_back(hex);
        eog.push_back(llama_vocab_is_eog(srv.vocab, t));
    }
    // ATK patch 3: the canvas as ONE detokenized string, with each position's byte offset into
    // it, so a client never has to join pieces itself; which positions hold a control token
    // (rendered empty in `text`, so the text either side of one is glued together); and the
    // per-step trajectory. `offsets` has canvas_length + 1 entries: the last is text.size().
    std::vector<llama_token> prefix;
    prefix.reserve(r.canvas0.size());
    json offsets = json::array(), control = json::array();
    for (const llama_token t : r.canvas0) {
        offsets.push_back((int64_t) common_detokenize(srv.vocab, prefix, false).size());
        prefix.push_back(t);
        control.push_back(llama_vocab_is_control(srv.vocab, t) && !llama_vocab_is_eog(srv.vocab, t));
    }
    const std::string text = common_detokenize(srv.vocab, prefix, false);
    offsets.push_back((int64_t) text.size());
    json traj = json::array();
    for (const auto & st : r.canvas0_trajectory) {
        json row{
            {"step", st.step}, {"cur_step", st.cur_step}, {"mean_entropy", st.mean_entropy},
            {"accepted", st.accepted}, {"flips", st.flips}, {"control", st.control},
            {"temp", st.temp}, {"ms", st.ms}, {"plateau", st.plateau},
            // ATK patch 4
            {"sprint", st.sprint}, {"eog_pos", st.eog_pos}, {"tail", st.tail}, {"osc_holds", st.osc_holds}, {"bound", st.bound},
        };
        if (!st.text.empty()) row["text"] = st.text;   // dumped with the replace handler, like the rest
        traj.push_back(row);
    }
    // Bytes that are not valid UTF-8 (a canvas can end inside a multi-byte character) must not
    // reach json::dump(), which would throw; such a string goes out as hex instead, flagged.
    bool text_utf8 = true;
    {
        const unsigned char * b = (const unsigned char *) text.data();
        size_t i = 0, n = text.size();
        while (i < n && text_utf8) {
            int len = 1;
            if      (b[i] < 0x80) len = 1;
            else if ((b[i] & 0xE0) == 0xC0) len = 2;
            else if ((b[i] & 0xF0) == 0xE0) len = 3;
            else if ((b[i] & 0xF8) == 0xF0) len = 4;
            else { text_utf8 = false; break; }
            if (i + len > n) { text_utf8 = false; break; }
            for (int k = 1; k < len; ++k) if ((b[i + k] & 0xC0) != 0x80) { text_utf8 = false; break; }
            i += len;
        }
    }
    std::string text_hex;
    if (!text_utf8) {
        text_hex.reserve(text.size() * 2);
        for (const unsigned char c : text) { text_hex.push_back(HEX[c >> 4]); text_hex.push_back(HEX[c & 15]); }
    }
    json out = json{
        {"tokens", r.canvas0}, {"pieces_hex", pieces}, {"eog", eog},
        {"entropy", r.canvas0_entropy}, {"flips", r.canvas0_flips},
        {"steps", r.canvas0_steps}, {"start_step", r.canvas0_start},
        // ATK patch 3
        {"offsets", offsets}, {"control", control}, {"control_count", r.canvas0_control},
        {"trajectory", traj}, {"plateaus", r.canvas0_plateaus},
        {"from_best", r.canvas0_from_best},
        {"best_entropy", r.canvas0_best_entropy}, {"last_entropy", r.canvas0_last_entropy},
        // ATK patch 4
        {"length", r.canvas0_length}, {"sprints", r.canvas0_sprints}, {"osc_holds", r.canvas0_osc_holds},
        {"eog_pos", r.canvas0_eog_pos}, {"tail_from", r.canvas0_tail_from},
        {"sc_cleared", r.canvas0_sc_cleared}, {"host_self_cond", r.canvas0_host_sc},
    };
    if (text_utf8) out["text"] = text; else out["text_hex"] = text_hex;
    // ATK patch 4: the self-conditioning readout, top-N per position (atk_readout_sc)
    if (r.sc_out_k > 0 && !r.sc_out_ids.empty()) {
        const int N = r.sc_out_k;
        const size_t L = r.sc_out_ids.size() / (size_t) N;
        json ids = json::array(), probs = json::array();
        for (size_t j = 0; j < L; ++j) {
            json ri = json::array(), rp = json::array();
            for (int k = 0; k < N; ++k) {
                if (r.sc_out_ids[j * N + k] < 0) break;
                ri.push_back(r.sc_out_ids[j * N + k]);
                rp.push_back(r.sc_out_probs[j * N + k]);
            }
            ids.push_back(ri); probs.push_back(rp);
        }
        out["self_cond"] = json{{"k", N}, {"ids", ids}, {"probs", probs}};
    }
    return out;
}

// ------------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::string hostname = "127.0.0.1";
    int         port     = 8080;
    std::string api_key;
    bool        enable_metrics = false;
    bool        enable_slots   = false;

    // Pull server-only flags out of argv before common_params_parse (which validates against
    // LLAMA_EXAMPLE_DIFFUSION and would reject these). Everything else (-m, --mmproj, -ngl, -c,
    // --seed, --top-k*, -t, ...) is forwarded and parsed normally.
    std::vector<char *> fwd;
    fwd.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * def) -> std::string { return (i + 1 < argc) ? argv[++i] : def; };
        if (a == "--host")            { hostname = next("127.0.0.1"); }
        else if (a == "--port")       { port = atoi(next("8080").c_str()); }
        else if (a == "--api-key")    { api_key = next(""); }
        else if (a == "--metrics")    { enable_metrics = true; }
        else if (a == "--slots")      { enable_slots = true; }
        else if (a == "--no-metrics") { enable_metrics = false; }
        else if (a == "--no-slots")   { enable_slots = false; }
        else                          { fwd.push_back(argv[i]); }
    }

    common_params params;
    params.diffusion.steps = DEF_MAX_DENOISE_STEPS;
    if (!common_params_parse((int) fwd.size(), fwd.data(), params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();
    if (params.diffusion.cuda_mmq_max_x < -1) {
        SRV_ERR("%s\n", "--diffusion-cuda-mmq-max-x must be -1, 0, or a positive value");
        return 1;
    }
    diffusion_configure_cuda_mmq_env(params.diffusion);
    if (!params.diffusion.device_self_cond && params.diffusion.fused_self_cond_embd) {
        SRV_WRN("%s\n", "disabling fused self-conditioning embedding because device self-conditioning is disabled");
        params.diffusion.fused_self_cond_embd = false;
    }
    params.diffusion.self_cond_top_k = diffusion_self_cond_top_k(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    SRV_INF("%s\n", "loading model");

    // ATK patch 4: build the loader params the way every other example does, from common's
    // converter, so EVERY flag common_params_parse accepts reaches the loader: -ot / --cpu-moe /
    // --n-cpu-moe (patch 2 passed these by hand), and now also --override-kv (e.g.
    // diffusion-gemma.expert_used_count=int:N), --mlock, --no-mmap, -sm / -ts / -mg,
    // --check-tensors. Before this, those were parsed and silently dropped. The one default kept
    // from the PR: without -ngl, offload every layer (999), not common's -1 "auto-fit".
    llama_model_params model_params = common_model_params_to_llama(params);
    if (params.n_gpu_layers < 0) {
        model_params.n_gpu_layers = 999;
    }

    diffusion_server srv;
    srv.model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!srv.model) {
        SRV_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(srv.model)) {
        SRV_ERR("'%s' is not a diffusion model\n", params.model.path.c_str());
        llama_model_free(srv.model);
        return 1;
    }

    srv.vocab         = llama_model_get_vocab(srv.model);
    srv.n_vocab       = llama_vocab_n_tokens(srv.vocab);
    srv.canvas_length = DEF_CANVAS_LENGTH;
    srv.n_steps       = std::max(params.diffusion.steps, 1);
    srv.diffusion     = params.diffusion;
    srv.n_ub          = srv.canvas_length;
    srv.n_ctx         = params.n_ctx > 0 ? (int) params.n_ctx : 4096;
    if (srv.n_ctx < 2 * srv.canvas_length) srv.n_ctx = 2 * srv.canvas_length;
    srv.topk_fixed    = (params.sampling.user_sampling_config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K)
                            ? params.sampling.top_k
                            : std::max(params.diffusion.default_top_k, 0);
    srv.topk_start    = params.diffusion.top_k_start;
    srv.topk_end      = params.diffusion.top_k_end;
    srv.topk_tail     = params.diffusion.top_k_tail_correction;
    srv.model_path    = params.model.path;
    srv.metrics.t_start = (int64_t) std::time(nullptr);

    {
        std::string p = params.model.path;
        size_t slash = p.find_last_of("/\\");
        srv.model_id = (slash == std::string::npos) ? p : p.substr(slash + 1);
        size_t dot = srv.model_id.rfind(".gguf");
        if (dot != std::string::npos) srv.model_id = srv.model_id.substr(0, dot);
        if (srv.model_id.empty()) srv.model_id = "diffusion-gemma";
    }

    // ATK patch 4: same for the context: -fa, -ctk / -ctv (KV cache quantization: VRAM freed here
    // is VRAM for experts), --no-kv-offload, -t / -tb, --no-op-offload (patch 2 passed this one by
    // hand) now all apply. Two things are then pinned to what the sampler relies on: the batch
    // is the canvas (one ubatch per denoising step), and the SWA cache is FULL-size -- common's
    // default is the window-size cache, but the prefix cache (patch 2) removes the canvas's K/V
    // and the tail of a changed prompt with llama_memory_seq_rm, which is exact only on a
    // full-size cache (llama's own default; the PR relied on it implicitly).
    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_ctx    = srv.n_ctx;
    ctx_params.n_batch  = srv.n_ub;
    ctx_params.n_ubatch = srv.n_ub;
    ctx_params.swa_full = true;
    ctx_params.diffusion_self_cond_top_k = srv.diffusion.self_cond_top_k;
    ctx_params.diffusion_input_gpu_groups = srv.diffusion.input_gpu_groups;
    ctx_params.diffusion_fused_self_cond_embd = srv.diffusion.fused_self_cond_embd;
    ctx_params.diffusion_fuse_final_logit_softcap = srv.diffusion.fuse_final_logit_softcap;
    ctx_params.diffusion_separate_encoder_decoder = srv.diffusion.separate_encoder_decoder;

    srv.ctx = llama_init_from_model(srv.model, ctx_params);
    if (!srv.ctx) {
        SRV_ERR("%s\n", "failed to create context");
        llama_model_free(srv.model);
        return 1;
    }
    llama_set_n_threads(srv.ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    srv.mem   = llama_get_memory(srv.ctx);
    srv.batch = llama_batch_init(srv.n_ub, 0, 1);
    srv.templates = common_chat_templates_init(srv.model, "");

    const int topk_max_requested =
        (srv.topk_start > 0 && srv.topk_end > 0) ? std::max(srv.topk_start, srv.topk_end) : srv.topk_fixed;
    const bool gpu_sampling_requested = params.diffusion.gpu_sampling;
    srv.use_gpu_sampling = gpu_sampling_requested &&
                           llama_diffusion_sample_topk_supported(srv.ctx);
    srv.use_device_self_cond = srv.use_gpu_sampling && params.diffusion.device_self_cond;
    srv.use_device_loop = srv.use_device_self_cond && params.diffusion.device_denoise_loop;
    srv.device_early_stop_interval =
        srv.use_device_loop && !params.diffusion.run_max_denoising_step ? 1 : 0;
    llama_set_diffusion_gpu_sampling(srv.ctx, srv.use_gpu_sampling);
    // ATK patch 4: the top-k rows are on the host whenever self-conditioning is not kept on the device
    srv.host_self_cond = !srv.use_device_self_cond;
    srv.init_atk_tables();
    SRV_INF("ATK patch 4: host self-cond %s | noise_text table: %zu of %d tokens | pad token: %d | canvas <= %d\n",
            srv.host_self_cond ? "ON (lab mode: margin/temp_pos_gain/rep_penalty/sc_sharpen/lateral_noise/readout_sc/sc seed/soft_hold available)"
                               : "off (launch with --no-diffusion-device-selfcond for the host self-cond features)",
            srv.noise_text_ids.size(), srv.n_vocab, srv.pad_token, srv.canvas_length);
    const bool default_topk_gpu_ok = topk_max_requested <= 0 || topk_max_requested <= GPU_SAMPLING_MAX_TOP_K;

    const uint32_t default_seed = params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed;

    SRV_INF("%s\n", llama_print_system_info());
    SRV_INF("model loaded: '%s' | n_ctx = %d | canvas = %d | denoise steps = %d | 1 slot\n",
            srv.model_id.c_str(), srv.n_ctx, srv.canvas_length, srv.n_steps);
    SRV_INF("gpu sampling: %s%s%s%s | top-k fixed=%d anneal=[%d->%d] tail_correction=%d\n",
            srv.use_gpu_sampling ? "on" : "off",
            (!default_topk_gpu_ok ? " (default top-k will use CPU fallback until k <= CUDA limit)" :
             (!gpu_sampling_requested ? " (disabled by --no-diffusion-gpu-sampling)" : "")),
            srv.use_device_self_cond ? " | device self-cond: on" : "",
            srv.use_device_loop ? " | device loop: on" : "",
            srv.topk_fixed, srv.topk_start, srv.topk_end, srv.topk_tail ? 1 : 0);
    if (srv.device_early_stop_interval > 0) {
        SRV_INF("device early-stop interval: %d\n", srv.device_early_stop_interval);
    } else if (srv.use_device_loop && params.diffusion.run_max_denoising_step) {
        SRV_INF("%s\n", "device early-stop polling disabled; running max denoising steps");
    }
    if (srv.diffusion.force_top_k > 0) {
        SRV_INF("forcing request top-k to %d via --diffusion-force-top-k\n", srv.diffusion.force_top_k);
    }

    // ------------------------------------------------------------------------------------------
    // HTTP server
    // ------------------------------------------------------------------------------------------
    httplib::Server http;
    http.set_default_headers({
        {"Server", "llama.cpp-diffusion-gemma"},
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
    });

    // access log (llama-server: SRV_TRC "done request: METHOD PATH ADDR STATUS"; skip noisy paths)
    http.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        if (req.path == "/health" || req.path == "/v1/health" || req.path == "/metrics" ||
            req.path == "/props"  || req.path == "/models"    || req.path == "/v1/models") {
            return;
        }
        SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
        SRV_DBG("request body: %s\n", req.body.c_str());
    });

    if (!api_key.empty()) {
        http.set_pre_routing_handler([api_key](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "OPTIONS" || req.path == "/health" || req.path == "/v1/health") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            if (req.get_header_value("Authorization") != "Bearer " + api_key) {
                res.status = 401;
                res.set_content(error_json("invalid api key", "authentication_error", 401).dump(), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    http.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

    auto health = [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    };
    http.Get("/health", health);
    http.Get("/v1/health", health);

    http.Get("/v1/models", [&srv](const httplib::Request &, httplib::Response & res) {
        json models = json::array();
        models.push_back({{"id", srv.model_id}, {"object", "model"}, {"created", (int64_t) std::time(nullptr)}, {"owned_by", "local"}});
        res.set_content(json{{"object", "list"}, {"data", models}}.dump(), "application/json");
    });
    http.Get("/models", [&srv](const httplib::Request &, httplib::Response & res) {
        json models = json::array();
        models.push_back({{"id", srv.model_id}, {"object", "model"}, {"created", (int64_t) std::time(nullptr)}, {"owned_by", "local"}});
        res.set_content(json{{"object", "list"}, {"data", models}}.dump(), "application/json");
    });

    // /props (llama-server analogue): server + model + default generation settings
    http.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        json props{
            {"default_generation_settings", {
                {"n_ctx", srv.n_ctx},
                {"canvas_length", srv.canvas_length},
                {"n_denoise_steps", srv.n_steps},
                {"temperature_min", TEMP_MIN},
                {"temperature_max", TEMP_MAX},
                {"entropy_bound", ENTROPY_BOUND},
                {"confidence_threshold", CONFIDENCE_THRESHOLD},
                {"stability_threshold", STABILITY_THRESHOLD},
                {"self_cond_topk", srv.diffusion.self_cond_top_k},
                {"gpu_sampling", srv.use_gpu_sampling},
                {"device_self_cond", srv.use_device_self_cond},
                {"device_loop", srv.use_device_loop},
                {"device_early_stop_interval", srv.device_early_stop_interval},
                {"run_max_denoising_step", srv.diffusion.run_max_denoising_step},
                {"fused_self_cond_embd", srv.diffusion.fused_self_cond_embd},
                {"fuse_final_logit_softcap", srv.diffusion.fuse_final_logit_softcap},
                {"cuda_direct_self_cond", srv.diffusion.cuda_direct_self_cond},
                {"cuda_final_tokens_on_stop", srv.diffusion.cuda_final_tokens_on_stop},
                {"cuda_fused_top_k_sample", srv.diffusion.cuda_fused_top_k_sample},
                {"cuda_parallel_full_softmax", srv.diffusion.cuda_parallel_full_softmax},
                {"cuda_fused_full_softmax", srv.diffusion.cuda_fused_full_softmax},
                {"cuda_mmq_max_x", srv.diffusion.cuda_mmq_max_x},
                {"top_k", srv.topk_fixed},
                {"top_k_start", srv.topk_start},
                {"top_k_end", srv.topk_end},
                {"top_k_tail_correction", srv.topk_tail},
            }},
            // ATK patch 2: what this build accepts beyond the PR's own server
            {"atk", {
                {"patch", 4}, {"canvas", true}, {"readout", true}, {"sampler", true}, {"prompt_cache", true},
                // ATK patch 4: what a client can rely on, and what needs the host self-cond launch flag
                {"host_self_cond", srv.host_self_cond},
                {"host_self_cond_features", json::array({"accept_order=margin", "temp_pos_gain", "rep_penalty", "sc_sharpen",
                                                         "lateral_noise", "atk_readout_sc", "atk_canvas.self_cond", "atk_canvas.soft_hold"})},
                {"canvas_length_max", srv.canvas_length},
                {"noise_text_tokens", (int64_t) srv.noise_text_ids.size()},
                {"pad_token", srv.pad_token},
                {"features", json::array({"canvas_length", "stability_steps", "eog_tail", "eog_tail_pad", "noise_text", "noise_clear_sc",
                                          "demolish_clear_sc", "sprint", "pos_bias", "accept_order", "bound_end", "temp_pos_gain",
                                          "rep_penalty", "sc_sharpen", "lateral_noise", "osc_hold", "n_candidates", "atk_readout_sc",
                                          "atk_canvas.self_cond", "atk_canvas.soft_hold", "atk_trajectory_text", "atk_stream_steps"})},
            }},
            {"total_slots", 1},
            {"model_path", srv.model_path},
            {"model_alias", srv.model_id},
            {"n_ctx", srv.n_ctx},
            {"build_info", srv.build_info},
            {"generation_type", "block-diffusion"},
            {"endpoint_slots", enable_slots},
            {"endpoint_metrics", enable_metrics},
        };
        res.set_content(props.dump(), "application/json");
    });

    // /metrics (Prometheus). Gated behind --metrics, like llama-server.
    if (enable_metrics) {
        http.Get("/metrics", [&srv](const httplib::Request &, httplib::Response & res) {
            server_metrics & m = srv.metrics;
            std::lock_guard<std::mutex> lk(m.mu);
            const int n_proc = m.n_processing.load();
            char buf[8192];
            int n = snprintf(buf, sizeof(buf),
                // --- counters (re-mapped to diffusion where the AR concept differs) ---
                "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
                "# TYPE llamacpp:prompt_tokens_total counter\n"
                "llamacpp:prompt_tokens_total %llu\n"
                "# HELP llamacpp:prompt_seconds_total Prompt (encoder prefill) process time.\n"
                "# TYPE llamacpp:prompt_seconds_total counter\n"
                "llamacpp:prompt_seconds_total %.3f\n"
                "# HELP llamacpp:tokens_predicted_total Number of answer tokens generated.\n"
                "# TYPE llamacpp:tokens_predicted_total counter\n"
                "llamacpp:tokens_predicted_total %llu\n"
                "# HELP llamacpp:tokens_predicted_seconds_total Denoising (generation) process time.\n"
                "# TYPE llamacpp:tokens_predicted_seconds_total counter\n"
                "llamacpp:tokens_predicted_seconds_total %.3f\n"
                "# HELP llamacpp:n_decode_total Total number of llama_decode() calls (prefill + denoise + commit).\n"
                "# TYPE llamacpp:n_decode_total counter\n"
                "llamacpp:n_decode_total %llu\n"
                "# HELP llamacpp:requests_total Total number of completed generation requests.\n"
                "# TYPE llamacpp:requests_total counter\n"
                "llamacpp:requests_total %llu\n"
                "# HELP llamacpp:diffusion_blocks_total Total number of canvas blocks denoised.\n"
                "# TYPE llamacpp:diffusion_blocks_total counter\n"
                "llamacpp:diffusion_blocks_total %llu\n"
                "# HELP llamacpp:diffusion_steps_total Total number of denoising steps.\n"
                "# TYPE llamacpp:diffusion_steps_total counter\n"
                "llamacpp:diffusion_steps_total %llu\n"
                "# HELP llamacpp:diffusion_canvas_tokens_total Total canvas tokens denoised (blocks * canvas_length).\n"
                "# TYPE llamacpp:diffusion_canvas_tokens_total counter\n"
                "llamacpp:diffusion_canvas_tokens_total %llu\n"
                // --- gauges (last request) ---
                "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.\n"
                "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
                "llamacpp:prompt_tokens_seconds %.3f\n"
                "# HELP llamacpp:predicted_tokens_seconds Average answer-token throughput in tokens/s.\n"
                "# TYPE llamacpp:predicted_tokens_seconds gauge\n"
                "llamacpp:predicted_tokens_seconds %.3f\n"
                "# HELP llamacpp:diffusion_steps_per_second Denoising steps per second (last request).\n"
                "# TYPE llamacpp:diffusion_steps_per_second gauge\n"
                "llamacpp:diffusion_steps_per_second %.3f\n"
                "# HELP llamacpp:diffusion_canvas_tokens_per_second Canvas tokens per second (last request).\n"
                "# TYPE llamacpp:diffusion_canvas_tokens_per_second gauge\n"
                "llamacpp:diffusion_canvas_tokens_per_second %.3f\n"
                "# HELP llamacpp:diffusion_ms_per_step Milliseconds per denoising step (last request).\n"
                "# TYPE llamacpp:diffusion_ms_per_step gauge\n"
                "llamacpp:diffusion_ms_per_step %.3f\n"
                "# HELP llamacpp:requests_processing Number of requests currently generating.\n"
                "# TYPE llamacpp:requests_processing gauge\n"
                "llamacpp:requests_processing %d\n",
                (unsigned long long) m.n_prompt_tokens_total,
                m.t_prompt_ms_total / 1000.0,
                (unsigned long long) m.n_tokens_predicted_total,
                m.t_predicted_ms_total / 1000.0,
                (unsigned long long) m.n_decode_total,
                (unsigned long long) m.n_requests_total,
                (unsigned long long) m.n_blocks_total,
                (unsigned long long) m.n_steps_total,
                (unsigned long long) m.n_canvas_tokens_total,
                m.last_prompt_tps,
                m.last_predicted_tps,
                m.last_steps_per_second,
                m.last_canvas_tps,
                m.last_ms_per_step,
                n_proc);
            res.set_header("Process-Start-Time-Unix", std::to_string(m.t_start));
            res.set_content(std::string(buf, n > 0 ? (size_t) n : 0), "text/plain; version=0.0.4");
        });
        SRV_INF("%s\n", "metrics endpoint enabled at /metrics");
    }

    // /slots (single slot). Gated behind --slots, like llama-server.
    if (enable_slots) {
        http.Get("/slots", [&srv](const httplib::Request &, httplib::Response & res) {
            json slot{
                {"id", 0},
                {"n_ctx", srv.n_ctx},
                {"is_processing", srv.metrics.n_processing.load() > 0},
                {"generation_type", "block-diffusion"},
                {"params", {
                    {"canvas_length", srv.canvas_length},
                    {"n_denoise_steps", srv.n_steps},
                    {"self_cond_topk", srv.diffusion.self_cond_top_k},
                    {"gpu_sampling", srv.use_gpu_sampling},
                    {"device_self_cond", srv.use_device_self_cond},
                    {"device_loop", srv.use_device_loop},
                    {"device_early_stop_interval", srv.device_early_stop_interval},
                    {"run_max_denoising_step", srv.diffusion.run_max_denoising_step},
                    {"fused_self_cond_embd", srv.diffusion.fused_self_cond_embd},
                    {"fuse_final_logit_softcap", srv.diffusion.fuse_final_logit_softcap},
                    {"cuda_direct_self_cond", srv.diffusion.cuda_direct_self_cond},
                    {"cuda_final_tokens_on_stop", srv.diffusion.cuda_final_tokens_on_stop},
                    {"cuda_fused_top_k_sample", srv.diffusion.cuda_fused_top_k_sample},
                    {"cuda_parallel_full_softmax", srv.diffusion.cuda_parallel_full_softmax},
                    {"cuda_fused_full_softmax", srv.diffusion.cuda_fused_full_softmax},
                    {"cuda_mmq_max_x", srv.diffusion.cuda_mmq_max_x},
                    {"top_k", srv.topk_fixed},
                    {"top_k_start", srv.topk_start},
                    {"top_k_end", srv.topk_end},
                    {"top_k_tail_correction", srv.topk_tail},
                }},
            };
            res.set_content(json::array({slot}).dump(), "application/json");
        });
        SRV_INF("%s\n", "slots endpoint enabled at /slots");
    }

    // shared: run one generation (under the slot mutex), update metrics, print timing log
    // ATK patch 4: best-of-N -- n_candidates runs with seeds seed, seed+1, ...; the prompt's KV is
    // reused across them, the canvas with the lowest best mean entropy is returned, and every
    // candidate's answer goes out in "atk_candidates" so a verifier can choose differently.
    auto run_for_body = [&srv, default_seed](const json & body, const std::vector<llama_token> & prompt_tokens,
                                             const diffusion_hooks * hooks = nullptr) {
        diffusion_request rq = request_from_body(body, srv, default_seed);
        srv.metrics.n_processing.fetch_add(1);
        std::unique_lock<std::mutex> lock(srv.gen_mutex);
        diffusion_result r;
        if (rq.n_candidates <= 1) {
            r = srv.generate(prompt_tokens, rq, hooks);
        } else {
            json cands = json::array();
            int best = -1; float best_ent = INFINITY;
            std::vector<diffusion_result> results;
            for (int c = 0; c < rq.n_candidates; ++c) {
                diffusion_request rc = rq;
                rc.seed = rq.seed + (uint32_t) c;
                results.push_back(srv.generate(prompt_tokens, rc, hooks));
                const diffusion_result & rr = results.back();
                if (!rr.ok) { r = rr; break; }
                const float ent = rr.has_readout ? rr.canvas0_best_entropy : INFINITY;
                cands.push_back(json{{"seed", rc.seed}, {"answer", rr.answer}, {"best_entropy", ent},
                                     {"last_entropy", rr.canvas0_last_entropy}, {"steps", rr.n_steps_total},
                                     {"hit_eog", rr.hit_eog}, {"from_best", rr.canvas0_from_best}});
                if (ent < best_ent) { best_ent = ent; best = c; }
                SRV_INF("candidate %d/%d (seed %u): best entropy %.4f, %d steps%s\n", c + 1, rq.n_candidates, rc.seed, ent, rr.n_steps_total,
                        best == c ? " <- best so far" : "");
            }
            if (best >= 0 && (results.empty() || results.back().ok)) {
                r = results[best];
                for (size_t c = 0; c < cands.size(); ++c) cands[c]["chosen"] = (int) c == best;
                r.candidates = cands;
            }
        }
        lock.unlock();
        srv.metrics.n_processing.fetch_sub(1);
        if (r.ok) { srv.metrics.add(r); log_timings(r); }
        return r;
    };

    // POST /v1/chat/completions
    http.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400; res.set_content(error_json("'messages' (array) is required", "invalid_request_error", 400).dump(), "application/json"); return;
        }
        std::string prompt;
        try { prompt = srv.format_messages(body["messages"]); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("failed to format messages: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        // the chat template emits no BOS, so add it at tokenization (add_special)
        const std::vector<llama_token> prompt_tokens = common_tokenize(srv.vocab, prompt, true, true);
        const bool stream = body.value("stream", false);

        const diffusion_result r = run_for_body(body, prompt_tokens);
        if (!r.ok) { res.status = 500; res.set_content(error_json(r.error, "server_error", 500).dump(), "application/json"); return; }

        if (!stream) {
            json out{
                {"id", gen_id("chatcmpl")}, {"object", "chat.completion"}, {"created", (int64_t) std::time(nullptr)}, {"model", srv.model_id},
                {"choices", json::array({ json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", r.answer}}}, {"finish_reason", "stop"}} })},
                {"usage", usage_json(r)},
                {"timings", timings_json(r)},
            };
            // ATK patch 4: the chat endpoint carries the same extras as /v1/completions
            if (r.has_readout) out["atk_canvas_out"] = readout_json(srv, r);
            if (!r.candidates.empty()) out["atk_candidates"] = r.candidates;
            // ATK patch 2: replace, never throw, on a reply that ends inside a UTF-8 character
            res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        const std::string id = gen_id("chatcmpl");
        const int64_t created = (int64_t) std::time(nullptr);
        const std::string model_id = srv.model_id;
        const std::string answer = r.answer;
        const json timings = timings_json(r);
        const json usage = usage_json(r);
        const bool include_usage = body.value("stream_options", json::object()).value("include_usage", false);
        res.set_chunked_content_provider("text/event-stream",
            [id, created, model_id, answer, timings, usage, include_usage](size_t, httplib::DataSink & sink) {
                auto send = [&](const json & c) { std::string s = "data: " + c.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n"; return sink.write(s.data(), s.size()); };
                auto chunk = [&](const json & delta, const char * finish, const json * extra) {
                    json c{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model_id},
                           {"choices", json::array({ json{{"index", 0}, {"delta", delta}, {"finish_reason", finish ? json(finish) : json(nullptr)}} })}};
                    if (extra) c["timings"] = *extra;
                    if (finish && include_usage) c["usage"] = usage;
                    return send(c);
                };
                if (!chunk(json{{"role", "assistant"}}, nullptr, nullptr)) return false;
                if (!chunk(json{{"content", answer}}, nullptr, nullptr))   return false;
                if (!chunk(json::object(), "stop", &timings))              return false;
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    // POST /v1/completions (plain prompt)
    http.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            res.status = 400; res.set_content(error_json("'prompt' (string) is required", "invalid_request_error", 400).dump(), "application/json"); return;
        }
        const std::string prompt = body["prompt"].get<std::string>();
        const std::vector<llama_token> prompt_tokens = common_tokenize(srv.vocab, prompt, true, true);

        // ATK patch: the finished answer carries what a caller needs to tell a cut-off reply from a
        // finished one, and the thought channel when the model wrote one.
        auto final_choice = [](const diffusion_result & r) {
            json c{{"index", 0}, {"text", r.answer}, {"finish_reason", r.hit_eog ? "stop" : "length"}};
            if (!r.thought.empty()) c["reasoning_content"] = r.thought;
            return c;
        };

        if (!body.value("stream", false)) {
            const diffusion_result r = run_for_body(body, prompt_tokens);
            if (!r.ok) { res.status = 500; res.set_content(error_json(r.error, "server_error", 500).dump(), "application/json"); return; }
            json out{
                {"id", gen_id("cmpl")}, {"object", "text_completion"}, {"created", (int64_t) std::time(nullptr)}, {"model", srv.model_id},
                {"choices", json::array({ final_choice(r) })},
                {"usage", usage_json(r)},
                {"timings", timings_json(r)},
            };
            if (r.has_readout) out["atk_canvas_out"] = readout_json(srv, r);
            if (!r.candidates.empty()) out["atk_candidates"] = r.candidates;   // ATK patch 4
            // ATK patch 2: a reply can end inside a UTF-8 character (a canvas boundary, a byte-
            // fallback token); dump() would throw on it, so such bytes become U+FFFD instead
            res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        // ATK patch: stream=true runs the generation INSIDE the response, so the client sees every
        // denoising step ("atk_progress") and every finished 256-token block ("atk_block": the answer
        // so far by the same rule as the final one, or in_thought while a thought channel is open, and
        // the thought once it has closed),
        // then the final answer as the one "choices" chunk. A client that disconnects stops the
        // generation at the next step.
        const std::string id = gen_id("cmpl");
        const int64_t created = (int64_t) std::time(nullptr);
        res.set_chunked_content_provider("text/event-stream",
            [&srv, &run_for_body, body, prompt_tokens, id, created, final_choice](size_t, httplib::DataSink & sink) {
                auto send = [&](const json & c) {
                    // ATK patch 2: replace, never throw, on a partial UTF-8 character (see above)
                    const std::string s = "data: " + c.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                diffusion_hooks hooks;
                hooks.on_step = [&](int block, int step, int n_steps) {
                    return send(json{{"atk_progress", {{"block", block}, {"step", step}, {"n_steps", n_steps}}}});
                };
                // ATK patch 4: the canvas after every denoising step (atk_stream_steps) -- the text as
                // the model currently reads it, its mean entropy, and per-position entropy so a client
                // can colour each span by confidence while it watches the answer resolve
                hooks.on_step_done = [&](int block, int step, const std::vector<llama_token> & argmax,
                                         const std::vector<float> & entropy, float mean_entropy, int accepted) {
                    return send(json{{"atk_step", {
                        {"block", block}, {"step", step},
                        {"text", common_detokenize(srv.vocab, argmax, false)},
                        {"mean_entropy", mean_entropy}, {"accepted", accepted}, {"entropy", entropy},
                    }}});
                };
                std::vector<llama_token> so_far;
                const llama_token chan_open  = diffusion_single_token(srv.vocab, "<|channel>");
                const llama_token chan_close = diffusion_single_token(srv.vocab, "<channel|>");
                hooks.on_block = [&](int block, const std::vector<llama_token> & toks) {
                    so_far.insert(so_far.end(), toks.begin(), toks.end());
                    const diffusion_answer a = diffusion_extract(srv.vocab, so_far, chan_open, chan_close);
                    json ev{
                        {"index", block},
                        {"in_thought", a.in_thought},
                        {"hit_eog", a.hit_eog},
                        {"answer_so_far", a.in_thought ? std::string() : common_detokenize(srv.vocab, a.answer, false)},
                    };
                    if (!a.thought.empty()) ev["thought"] = common_detokenize(srv.vocab, a.thought, false);
                    return send(json{{"atk_block", ev}});
                };
                const diffusion_result r = run_for_body(body, prompt_tokens, &hooks);
                if (r.cancelled) {
                    SRV_INF("%s\n", "generation stopped: the client went away");
                    return false;
                }
                if (!r.ok) {
                    send(error_json(r.error, "server_error", 500));
                } else {
                    json fin{{"id", id}, {"object", "text_completion"}, {"created", created}, {"model", srv.model_id},
                             {"choices", json::array({ final_choice(r) })},
                             {"usage", usage_json(r)}, {"timings", timings_json(r)}};
                    if (r.has_readout) fin["atk_canvas_out"] = readout_json(srv, r);
                    if (!r.candidates.empty()) fin["atk_candidates"] = r.candidates;   // ATK patch 4
                    send(fin);
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    // ATK patch: POST /tokenize, as llama-server has it -- {"content": str, "add_special": bool,
    // "parse_special": bool} -> {"tokens": [...]}. Lets a client budget its context exactly.
    http.Post("/tokenize", [&srv](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & e) { res.status = 400; res.set_content(error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error", 400).dump(), "application/json"); return; }
        const std::string content = body.value("content", std::string());
        const bool add_special   = body.value("add_special", false);
        const bool parse_special = body.value("parse_special", true);
        const std::vector<llama_token> toks = common_tokenize(srv.vocab, content, add_special, parse_special);
        res.set_content(json{{"tokens", toks}}.dump(), "application/json");
    });

    SRV_INF("server is listening on http://%s:%d\n", hostname.c_str(), port);
    SRV_INF("%s\n", "all slots are idle");
    if (!http.listen(hostname, port)) {
        SRV_ERR("failed to bind to %s:%d\n", hostname.c_str(), port);
        llama_batch_free(srv.batch);
        llama_free(srv.ctx);
        llama_model_free(srv.model);
        llama_backend_free();
        return 1;
    }

    llama_batch_free(srv.batch);
    llama_free(srv.ctx);
    llama_model_free(srv.model);
    llama_backend_free();
    return 0;
}
