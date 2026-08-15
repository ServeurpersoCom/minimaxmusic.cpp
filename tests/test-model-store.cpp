// test-model-store.cpp: exercise ModelStore with real GGUF loads.
//
// Runs the store in both policies and prints what's resident at each step.
// Serves as living documentation of the expected require / release flow
// and catches regressions in eviction, refcounting or group logic.
// Uses the small modules only (cond, vae, depth); the LM and the DiT
// follow the exact same code path.
//
// Usage:
//   ./test-model-store --models <dir>
//
// Scans the registry exactly like the CLI binaries and picks the first
// entry of each bucket.

#include "model-registry.h"
#include "model-store.h"

#include <cstdio>
#include <cstring>
#include <string>

static void dump(ModelStore * s, const char * tag) {
    fprintf(stderr, "[Test] %s: modules=%d, vram=%.1f MB\n", tag, store_gpu_module_count(s),
            (float) store_vram_bytes(s) / (1024.0f * 1024.0f));
}

static bool expect_count(ModelStore * s, int want, const char * tag) {
    if (store_gpu_module_count(s) != want) {
        fprintf(stderr, "[Test] FAIL: %s: expected %d modules, got %d\n", tag, want, store_gpu_module_count(s));
        return false;
    }
    return true;
}

// Scenario 1: STRICT policy. Same-group modules coexist, release unloads
// on the spot, a require from the other group finds an empty map.
static int scenario_strict(const char * cond_path, const char * vae_path, const char * depth_path) {
    fprintf(stderr, "[Test] scenario 1: STRICT\n");
    ModelStore * s = store_create(EVICT_STRICT);
    dump(s, "empty");

    ModelKey  k_cond = { MODEL_COND, cond_path, 0, 0 };
    CondEnc * cond   = store_require_cond(s, k_cond);
    if (!cond) {
        fprintf(stderr, "[Test] FAIL: Cond load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require Cond");

    // VAE is in the same SYNTH group: both stay resident
    ModelKey  k_vae = { MODEL_VAE, vae_path, 0, 0 };
    FlowVAE * vae   = store_require_vae(s, k_vae);
    if (!vae) {
        fprintf(stderr, "[Test] FAIL: VAE load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require VAE (same group)");
    if (!expect_count(s, 2, "SYNTH group coexists")) {
        store_free(s);
        return 1;
    }

    store_release(s, cond);
    store_release(s, vae);
    dump(s, "after release both");
    if (!expect_count(s, 0, "STRICT unloads on release")) {
        store_free(s);
        return 1;
    }

    // AR group member loads into an empty map
    ModelKey       k_depth = { MODEL_DEPTH, depth_path, 0, 0 };
    DepthDecoder * depth   = store_require_depth(s, k_depth);
    if (!depth) {
        fprintf(stderr, "[Test] FAIL: Depth load\n");
        store_free(s);
        return 1;
    }
    dump(s, "after require Depth");
    if (!expect_count(s, 1, "AR group alone")) {
        store_free(s);
        return 1;
    }
    store_release(s, depth);
    if (!expect_count(s, 0, "empty after release Depth")) {
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 1: OK\n");
    return 0;
}

// Scenario 2: NEVER policy. Release keeps everything resident, a second
// require of the same key is a cache hit on the same pointer.
static int scenario_never(const char * cond_path, const char * vae_path) {
    fprintf(stderr, "[Test] scenario 2: NEVER\n");
    ModelStore * s = store_create(EVICT_NEVER);

    ModelKey  k_cond = { MODEL_COND, cond_path, 0, 0 };
    CondEnc * cond   = store_require_cond(s, k_cond);
    if (!cond) {
        fprintf(stderr, "[Test] FAIL: Cond load\n");
        store_free(s);
        return 1;
    }
    store_release(s, cond);
    dump(s, "after release Cond");
    if (!expect_count(s, 1, "NEVER keeps on release")) {
        store_free(s);
        return 1;
    }

    ModelKey  k_vae = { MODEL_VAE, vae_path, 0, 0 };
    FlowVAE * vae   = store_require_vae(s, k_vae);
    if (!vae) {
        fprintf(stderr, "[Test] FAIL: VAE load\n");
        store_free(s);
        return 1;
    }
    store_release(s, vae);
    dump(s, "after release VAE");
    if (!expect_count(s, 2, "NEVER accumulates")) {
        store_free(s);
        return 1;
    }

    CondEnc * cond2 = store_require_cond(s, k_cond);
    if (cond2 != cond) {
        fprintf(stderr, "[Test] FAIL: cache hit returned a different instance\n");
        store_free(s);
        return 1;
    }
    store_release(s, cond2);
    dump(s, "after cache hit + release");
    if (!expect_count(s, 2, "cache hit loads nothing")) {
        store_free(s);
        return 1;
    }

    store_free(s);
    fprintf(stderr, "[Test] scenario 2: OK\n");
    return 0;
}

int main(int argc, char ** argv) {
    const char * models = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models = argv[++i];
        }
    }
    if (!models) {
        fprintf(stderr, "Usage: %s --models <dir>\n", argv[0]);
        return 1;
    }

    ModelRegistry reg;
    if (!registry_scan(&reg, models)) {
        fprintf(stderr, "[Test] FAIL: cannot scan %s\n", models);
        return 1;
    }
    if (reg.cond.empty() || reg.vae.empty() || reg.depth.empty()) {
        fprintf(stderr, "[Test] FAIL: registry needs at least one cond, vae and depth GGUF\n");
        return 1;
    }
    std::string cond_path  = reg.cond.front().path;
    std::string vae_path   = reg.vae.front().path;
    std::string depth_path = reg.depth.front().path;

    if (scenario_strict(cond_path.c_str(), vae_path.c_str(), depth_path.c_str())) {
        return 1;
    }
    if (scenario_never(cond_path.c_str(), vae_path.c_str())) {
        return 1;
    }
    fprintf(stderr, "[Test] PASS\n");
    return 0;
}
