#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one coexistence group resident
//       at a time. The LM weights and the DiT weights never coexist by
//       construction, because they live in different groups.
//
//   coexistence groups
//       The pipeline interleaves some modules at a granularity where
//       evicting between them would thrash, so eviction operates on
//       groups, not single modules:
//         AR    = { LM, DEPTH }      interleaved per frame
//         SYNTH = { COND, DIT, VAE } interleaved per window and per song
//       Modules in the same group coexist freely. A require from another
//       group evicts every idle module of the resident group.
//
//   invariant held under BOTH policies
//       Exactly ONE instance per ModelKey for the whole process. Two
//       requires with the same key return the same pointer, so the AR
//       stage of pipeline_generate and pipeline_lm_generate share one LM.
//
// A ModelStore holds the GGML module instances that the pipeline needs
// (Qwen3 LM, depth decoder, condition encoder, DiT, flow VAE). The
// pipeline asks the store for a module by key and returns it when done.
// The store decides what stays in VRAM and what gets evicted, following
// the policy above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path, extras). Two requires
//   with the same key return the same instance. The LM extras (max_seq,
//   n_kv_sets) are process-lifetime constants from MM3PipelineParams; a
//   drift between two callers would silently duplicate 17 GB of weights.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 in another group cannot be
//   evicted: a conflicting require is a programming error (aborts). This
//   catches accidental overlap between groups that must not coexist.
//
// Threading
//   The store has no lock: a single worker thread owns it. The server
//   serializes GPU work on one worker; HTTP handlers never touch the
//   store. Adding a second worker requires adding a mutex here first.

#include "bpe.h"
#include "cond-enc.h"
#include "depth-decoder.h"
#include "dit.h"
#include "qwen3-lm.h"
#include "vae.h"

#include <cstddef>
#include <string>

struct ModelStore;

enum ModelKind {
    MODEL_LM,     // Qwen3LM       from MiniMax-Music3-language_model-*.gguf
    MODEL_DEPTH,  // DepthDecoder  from MiniMax-Music3-rvq_depth_decoder-*.gguf
    MODEL_COND,   // CondEnc       from MiniMax-Music3-condition_encoder-*.gguf
    MODEL_DIT,    // DiT           from MiniMax-Music3-transformer-*.gguf
    MODEL_VAE,    // FlowVAE       from MiniMax-Music3-vocoder-*.gguf
};

struct ModelKey {
    ModelKind   kind;
    std::string path;  // GGUF path the module is loaded from
    // LM-only extras (ignored for other kinds):
    int         max_seq;    // KV cache length, 0 = model context
    int         n_kv_sets;  // number of KV sets (2 * max_batch with CFG)
};

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one coexistence group resident
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

ModelStore * store_create(EvictPolicy policy);
void         store_free(ModelStore * s);

// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every module outside
// its coexistence group whose refcount is zero; if any conflicting module
// has refcount > 0 the store aborts (a programming error in the caller).
Qwen3LM *      store_require_lm(ModelStore * s, const ModelKey & k);
DepthDecoder * store_require_depth(ModelStore * s, const ModelKey & k);
CondEnc *      store_require_cond(ModelStore * s, const ModelKey & k);
DiT *          store_require_dit(ModelStore * s, const ModelKey & k);
FlowVAE *      store_require_vae(ModelStore * s, const ModelKey & k);

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void store_release(ModelStore * s, void * handle);

// CPU-resident accessor. Loaded on first call, kept forever, never
// evicted: the tokenizer travels with the LM GGUF metadata (a few MB).
// Returns NULL on load failure.
BPETokenizer * store_bpe(ModelStore * s, const char * lm_path);

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules. Used by test-model-store to assert
// eviction policy invariants.
size_t store_vram_bytes(const ModelStore * s);
int    store_gpu_module_count(const ModelStore * s);

// RAII helper. Builds on top of store_release, nothing else.
struct ModelHandle {
    ModelStore * store;
    void *       ptr;

    ModelHandle(ModelStore * s, void * p) : store(s), ptr(p) {}

    ~ModelHandle() {
        if (store && ptr) {
            store_release(store, ptr);
        }
    }

    // non-copyable, movable
    ModelHandle(const ModelHandle &)             = delete;
    ModelHandle & operator=(const ModelHandle &) = delete;

    ModelHandle(ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};
