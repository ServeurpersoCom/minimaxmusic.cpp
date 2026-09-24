#pragma once
// adapter.h: MiniMax Music 3 adapters merged into the weights at load.
//
// An adapter changes the planner LM, the flow DiT, or both. The two share no
// weight and live in different load groups, so each takes its own list of
// adapters with its own strength and reloads only when that list changes.
//
// The merge runs between the GGUF loads and wctx_alloc, on the staged
// PendingCopy of every projection, before the LM's QKV and gate/up fusion:
// fusion then concatenates already adapted rows. Every contribution to one
// tensor, from every adapter of the list, is summed in a single backend graph
// and the tensor is encoded back to its GGUF type once, so stacking adapters
// on a quantized base costs one requantization, not one per adapter.
//
// Formats, all normalized to the GGUF tensor names:
//
//   LM, PEFT (SimpleTuner, HOT-Step ace-train mm3-lm-train)
//       language_model.model.layers.N.self_attn.q_proj.lora_A.weight
//       base_model.model.model.layers.N.mlp.up_proj.lora_B.default.weight
//   LM, ComfyUI (ntc-ai sliders)
//       text_encoders.model.layers.N.self_attn.k_proj.lora_A.weight, .alpha
//   LM, LyCORIS LoKr (HOT-Step)
//       lycoris_layers_N_self_attn_q_proj.lokr_w1 | lokr_w2 | lokr_w2_a | lokr_w2_b
//   DiT, diffusers / PEFT (SimpleTuner)
//       transformer.transformer_blocks.N.attn.to_q.lora_A.weight
//   DiT, ComfyUI (SimpleTuner's default export)
//       diffusion_model.diffusion_transformer.transformer.layers.N.self_attn.to_qkv.lora_A.weight
//       ...layers.N.ff.ff.0.proj, ...layers.N.ff.ff.2, ...transformer.project_in
//
// A fused to_qkv, qkv_proj or gate_up_proj factor is split back onto the
// separate GGUF projections by rows: a delta B @ A restricted to rows [r0, r1)
// is B[r0:r1] @ A, exact for a shared A and for block diagonal fusions alike.
// A LoKr delta kron(w1, W2) restricted to whole blocks of W2 rows is
// kron(w1[l0:l1], W2).
//
// The DiT's ff_in emits value then gate. SimpleTuner records the order a
// diffusers checkpoint was trained in as swiglu_gate_first in __metadata__; a
// gate first LoRA has the two halves of its B rows swapped at merge, or the
// audio renders and is quietly wrong.
//
// Operations per tensor:
//   LoRA   W += s * (alpha / rank) * B @ A      alpha from a per tensor
//                                               .alpha, then __metadata__
//                                               alpha, then adapter_config.json,
//                                               else rank
//   LoKr   W += s * (alpha / dim) * kron(w1, W2), scale 1 for a monolithic w2
//   diff   W += s * D                           ComfyUI .diff
//
// Parameterisations whose delta is not B @ A - DoRA, LoHa, HiRA, a PiSSA
// delta tied to a residual base - are refused by name rather than merged as
// if they were plain LoRA. Keys that name nothing in the model (artist
// tokens, KV prefixes, conditioning branches) are counted and reported.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "safetensors.h"
#include "weight-ctx.h"
#include "yyjson.h"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#    ifndef S_ISDIR
#        define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#    endif
#else
#    include <dirent.h>
#endif

enum AdapterHalf {
    ADAPTER_LM  = 0,
    ADAPTER_DIT = 1,
};

// One adapter bound to one half: where it lives and how strongly it applies.
// The pipeline builds one list per half, holding only the adapters that
// touch that half with a non zero scale, and the model store keys the half
// on that list.
struct AdapterSpec {
    std::string path;   // a .safetensors file or a directory of them
    float       scale;  // user strength for this half
};

static std::vector<std::string> adapter_files(const std::string & path);

// Stable text of a spec list, the model store key of an adapted half; the
// size and time of each file are in it, so a file replaced in place is merged
// again rather than served from the store
static std::string adapter_signature(const std::vector<AdapterSpec> & specs) {
    std::string sig;
    for (const auto & s : specs) {
        char scale[32];
        snprintf(scale, sizeof(scale), "%.6g", (double) s.scale);
        sig += s.path + "@" + scale;
        for (const auto & file : adapter_files(s.path)) {
            std::error_code ec;
            auto            fp   = std::filesystem::u8path(file);
            auto            size = std::filesystem::file_size(fp, ec);
            auto            time = std::filesystem::last_write_time(fp, ec).time_since_epoch().count();
            sig += "|" + std::to_string((unsigned long long) size) + ":" + std::to_string((long long) time);
        }
        sig += ";";
    }
    return sig;
}

// Rows of a fused site that map onto one GGUF projection
enum AdapterSlice {
    SLICE_NONE,
    SLICE_Q,
    SLICE_K,
    SLICE_V,
    SLICE_GATE,
    SLICE_UP,
};

// What a key contributes
enum AdapterRole {
    ROLE_LORA_A,
    ROLE_LORA_B,
    ROLE_ALPHA,
    ROLE_LOKR_W1,
    ROLE_LOKR_W1A,
    ROLE_LOKR_W1B,
    ROLE_LOKR_W2,
    ROLE_LOKR_W2A,
    ROLE_LOKR_W2B,
    ROLE_DIFF,
    ROLE_FULL,
};

struct AdapterKey {
    AdapterHalf half;
    std::string module;  // GGUF module, "model.layers.3.mlp.up_proj", "transformer_blocks.7.attn.to_q"
    bool        bias;    // targets module.bias instead of module.weight
    bool        fused;   // to_qkv, qkv_proj or gate_up_proj, split by rows at merge
    AdapterRole role;
};

static bool adapter_ends_with(const std::string & s, const char * suffix, std::string * head) {
    size_t n = strlen(suffix);
    if (s.size() <= n || s.compare(s.size() - n, n, suffix) != 0) {
        return false;
    }
    *head = s.substr(0, s.size() - n);
    return true;
}

static bool adapter_starts_with(const std::string & s, const char * prefix, std::string * tail) {
    size_t n = strlen(prefix);
    if (s.size() < n || s.compare(0, n, prefix) != 0) {
        return false;
    }
    *tail = s.substr(n);
    return true;
}

static bool adapter_is_layer_index(const std::string & s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

// Splits "<index>.<tail>" after a block prefix
static bool adapter_split_index(const std::string & s, std::string * index, std::string * tail) {
    size_t dot = s.find('.');
    if (dot == std::string::npos || !adapter_is_layer_index(s.substr(0, dot))) {
        return false;
    }
    *index = s.substr(0, dot);
    *tail  = s.substr(dot + 1);
    return true;
}

// The LM projections an adapter may target
static bool adapter_lm_site(const std::string & site) {
    static const char * sites[] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",   "self_attn.o_proj", "mlp.gate_proj",
        "mlp.up_proj",      "mlp.down_proj",    "self_attn.qkv_proj", "mlp.gate_up_proj",
    };
    for (const char * s : sites) {
        if (site == s) {
            return true;
        }
    }
    return false;
}

// A DiT block tail in the ComfyUI naming to the GGUF (diffusers) naming
static bool adapter_dit_comfy_site(const std::string & tail, std::string * site) {
    static const char * map[][2] = {
        { "self_attn.to_qkv", "attn.to_qkv"   },
        { "self_attn.to_out", "attn.to_out.0" },
        { "ff.ff.0.proj",     "ff_in"         },
        { "ff.ff.2",          "ff_out"        },
    };
    for (const auto & m : map) {
        if (tail == m[0]) {
            *site = m[1];
            return true;
        }
    }
    return false;
}

static bool adapter_dit_site(const std::string & site) {
    static const char * sites[] = {
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_qkv", "attn.to_out.0", "ff_in", "ff_out",
    };
    for (const char * s : sites) {
        if (site == s) {
            return true;
        }
    }
    return false;
}

// Normalizes one safetensors key. Returns false for a key that names nothing
// the model has.
static bool adapter_parse_key(const std::string & raw, AdapterKey * out) {
    std::string key = raw;
    std::string rest;

    // role suffix first, the rest of the key is the module
    static const struct {
        const char * suffix;
        AdapterRole  role;
    } roles[] = {
        { ".lora_A.default.weight", ROLE_LORA_A   },
        { ".lora_B.default.weight", ROLE_LORA_B   },
        { ".lora_A.weight",         ROLE_LORA_A   },
        { ".lora_B.weight",         ROLE_LORA_B   },
        { ".lora_down.weight",      ROLE_LORA_A   },
        { ".lora_up.weight",        ROLE_LORA_B   },
        { ".lora.down.weight",      ROLE_LORA_A   },
        { ".lora.up.weight",        ROLE_LORA_B   },
        { ".lora_A",                ROLE_LORA_A   },
        { ".lora_B",                ROLE_LORA_B   },
        { ".lora_alpha",            ROLE_ALPHA    },
        { ".alpha",                 ROLE_ALPHA    },
        { ".lokr_w2_a",             ROLE_LOKR_W2A },
        { ".lokr_w2_b",             ROLE_LOKR_W2B },
        { ".lokr_w1_a",             ROLE_LOKR_W1A },
        { ".lokr_w1_b",             ROLE_LOKR_W1B },
        { ".lokr_w1",               ROLE_LOKR_W1  },
        { ".lokr_w2",               ROLE_LOKR_W2  },
        { ".diff",                  ROLE_DIFF     },
    };

    bool found = false;
    for (const auto & r : roles) {
        if (adapter_ends_with(key, r.suffix, &rest)) {
            out->role = r.role;
            key       = rest;
            found     = true;
            break;
        }
    }
    if (!found) {
        return false;
    }

    // PEFT wraps a module it replaced
    for (const char * infix : { ".original_module.", ".base_layer." }) {
        size_t p = key.find(infix);
        if (p != std::string::npos) {
            key = key.substr(0, p) + "." + key.substr(p + strlen(infix));
        }
    }

    // container prefixes, some of them saying which half the file meant
    int hint = -1;
    for (const char * pfx : { "model.diffusion_model.", "diffusion_model." }) {
        if (adapter_starts_with(key, pfx, &rest)) {
            hint = ADAPTER_DIT;
            key  = rest;
            break;
        }
    }
    if (adapter_starts_with(key, "base_model.model.", &rest)) {
        key = rest;
    }
    if (adapter_starts_with(key, "language_model.", &rest) || adapter_starts_with(key, "text_encoders.", &rest)) {
        // ComfyUI files the planner LM under its text encoders
        hint = ADAPTER_LM;
        key  = rest;
    } else if (adapter_starts_with(key, "transformer.", &rest) && hint != ADAPTER_LM) {
        hint = ADAPTER_DIT;
        key  = rest;
    }

    out->bias = false;

    // LyCORIS LoKr of the LM: lycoris_layers_<N>_<site with dots as underscores>
    if (adapter_starts_with(key, "lycoris_layers_", &rest)) {
        size_t us = rest.find('_');
        if (us == std::string::npos || !adapter_is_layer_index(rest.substr(0, us))) {
            return false;
        }
        std::string flat = rest.substr(us + 1);
        std::string site;
        for (const char * s : { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) {
            std::string u = s;
            std::replace(u.begin(), u.end(), '.', '_');
            if (flat == u) {
                site = s;
                break;
            }
        }
        if (site.empty()) {
            return false;
        }
        out->half   = ADAPTER_LM;
        out->module = "model.layers." + rest.substr(0, us) + "." + site;
        out->fused  = false;
        return true;
    }

    std::string index, tail;

    // DiT, ComfyUI naming
    if (adapter_starts_with(key, "diffusion_transformer.", &rest)) {
        std::string blocks;
        if (adapter_starts_with(rest, "transformer.layers.", &blocks) && adapter_split_index(blocks, &index, &tail)) {
            std::string site;
            if (!adapter_dit_comfy_site(tail, &site)) {
                return false;
            }
            out->half   = ADAPTER_DIT;
            out->module = "transformer_blocks." + index + "." + site;
            out->fused  = site == "attn.to_qkv";
            return true;
        }
        if (rest == "transformer.project_in" || rest == "transformer.project_out") {
            out->half   = ADAPTER_DIT;
            out->module = rest == "transformer.project_in" ? "proj_in" : "proj_out";
            out->fused  = false;
            return true;
        }
        return false;
    }

    // DiT, diffusers naming
    if (adapter_starts_with(key, "transformer_blocks.", &rest) && adapter_split_index(rest, &index, &tail)) {
        if (hint == ADAPTER_LM || !adapter_dit_site(tail)) {
            return false;
        }
        out->half   = ADAPTER_DIT;
        out->module = "transformer_blocks." + index + "." + tail;
        out->fused  = tail == "attn.to_qkv";
        return true;
    }
    if ((key == "proj_in" || key == "proj_out") && hint != ADAPTER_LM) {
        out->half   = ADAPTER_DIT;
        out->module = key;
        out->fused  = false;
        return true;
    }

    // LM, model.layers.N.<site>
    if (adapter_starts_with(key, "layers.", &rest)) {
        key = "model." + key;
    }
    if (!adapter_starts_with(key, "model.layers.", &rest) || !adapter_split_index(rest, &index, &tail)) {
        return false;
    }
    if (hint == ADAPTER_DIT || !adapter_lm_site(tail)) {
        return false;
    }
    out->half   = ADAPTER_LM;
    out->module = "model.layers." + index + "." + tail;
    out->fused  = tail == "self_attn.qkv_proj" || tail == "mlp.gate_up_proj";
    return true;
}

// Safetensors tensor to F32. F32, BF16 and F16 payloads.
static bool adapter_to_f32(const void * src, float * dst, int64_t n, const std::string & dtype) {
    if (dtype == "F32") {
        memcpy(dst, src, (size_t) n * sizeof(float));
    } else if (dtype == "BF16") {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
    } else {
        return false;
    }
    return true;
}

static int64_t adapter_numel(const STEntry & e) {
    int64_t n = 1;
    for (int i = 0; i < e.n_dims; i++) {
        n *= e.shape[i];
    }
    return n;
}

// __metadata__ of a safetensors file, every value kept as text
static std::map<std::string, std::string> adapter_read_metadata(const STFile & st) {
    std::map<std::string, std::string> out;
    if (st.data_offset <= 8) {
        return out;
    }
    yyjson_doc * doc = yyjson_read((const char *) st.mapping + 8, st.data_offset - 8, 0);
    if (!doc) {
        return out;
    }
    yyjson_val * meta = yyjson_obj_get(yyjson_doc_get_root(doc), "__metadata__");
    if (meta && yyjson_is_obj(meta)) {
        size_t       idx, max;
        yyjson_val * k;
        yyjson_val * v;
        yyjson_obj_foreach(meta, idx, max, k, v) {
            if (yyjson_is_str(v)) {
                out[yyjson_get_str(k)] = std::string(yyjson_get_str(v), yyjson_get_len(v));
            }
        }
    }
    yyjson_doc_free(doc);
    return out;
}

// What an adapter_config.json beside the weights says about the scale: its
// lora_alpha (or alpha), 0 when absent, and whether it trained rank-stabilised
// (use_rslora), whose scale is alpha / sqrt(rank) instead of alpha / rank
struct AdapterConfig {
    float       alpha  = 0.0f;
    bool        rslora = false;
    std::string refused;  // a setting the merge cannot honour exactly
};

static AdapterConfig adapter_read_config(const std::string & dir) {
    AdapterConfig config;
    std::string   path = dir + "/adapter_config.json";
    yyjson_doc *  doc  = yyjson_read_file(path.c_str(), 0, NULL, NULL);
    if (!doc) {
        return config;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    for (const char * key : { "lora_alpha", "alpha" }) {
        yyjson_val * v = yyjson_obj_get(root, key);
        if (v && yyjson_is_num(v)) {
            config.alpha = (float) yyjson_get_num(v);
            break;
        }
    }
    yyjson_val * rs = yyjson_obj_get(root, "use_rslora");
    config.rslora   = rs && yyjson_is_bool(rs) && yyjson_get_bool(rs);
    // per module ranks and alphas would each need their own scale
    for (const char * key : { "rank_pattern", "alpha_pattern" }) {
        yyjson_val * v = yyjson_obj_get(root, key);
        if (v && yyjson_is_obj(v) && yyjson_obj_size(v) > 0) {
            config.refused = std::string(key) + " in adapter_config.json is not supported";
        }
    }
    yyjson_doc_free(doc);
    return config;
}

static bool adapter_is_dir(const std::string & path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

static bool adapter_is_file(const std::string & path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && !S_ISDIR(sb.st_mode);
}

// Names of the entries of a directory, files or subdirectories, sorted
static std::vector<std::string> adapter_list_dir(const std::string & dir, bool want_dirs) {
    std::vector<std::string> out;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return out;
    }
    do {
        std::string name = fd.cFileName;
        if (name == "." || name == "..") {
            continue;
        }
        bool is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (is_dir == want_dirs) {
            out.push_back(name);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return out;
    }
    while (struct dirent * e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        if (adapter_is_dir(dir + "/" + name) == want_dirs) {
            out.push_back(name);
        }
    }
    closedir(d);
#endif
    std::sort(out.begin(), out.end());
    return out;
}

static bool adapter_is_safetensors(const std::string & name) {
    std::string head;
    return adapter_ends_with(name, ".safetensors", &head);
}

// The weight files of an adapter: the file itself, or in a directory its
// PEFT adapter_model.safetensors, else every .safetensors it holds
static std::vector<std::string> adapter_files(const std::string & path) {
    std::vector<std::string> files;
    if (!adapter_is_dir(path)) {
        files.push_back(path);
        return files;
    }
    std::string peft = path + "/adapter_model.safetensors";
    if (adapter_is_file(peft)) {
        files.push_back(peft);
        return files;
    }
    for (const auto & name : adapter_list_dir(path, false)) {
        if (adapter_is_safetensors(name)) {
            files.push_back(path + "/" + name);
        }
    }
    return files;
}

// A parameterisation whose delta is not B @ A (or kron) cannot be merged as
// one; empty when the file is plain LoRA, LoKr or a diff.
static std::string adapter_unsupported(const STFile & st, const std::map<std::string, std::string> & meta) {
    for (const auto & e : st.entries) {
        std::string head;
        if (adapter_ends_with(e.name, ".lora_magnitude_vector", &head) ||
            adapter_ends_with(e.name, ".lora_magnitude_vector.weight", &head) ||
            adapter_ends_with(e.name, ".dora_scale", &head)) {
            return "DoRA adapters are not supported, only LoRA and LoKr";
        }
        if (e.name.find(".hada_w") != std::string::npos) {
            return "LoHa adapters are not supported, only LoRA and LoKr";
        }
        if (e.name == "hot_step.param_method" && e.dtype == "F32" && adapter_numel(e) == 1) {
            float method = 0.0f;
            memcpy(&method, st_data(st, e), sizeof(float));
            if (method != 0.0f) {
                return "this adapter applies as DoRA, HiRA, LoHa or a PiSSA delta, not as plain LoRA";
            }
        }
    }
    auto it = meta.find("hot_step_pissa_delta");
    if (it != meta.end()) {
        return "a PiSSA delta adapter needs its residual base; export it as plain LoRA";
    }
    return {};
}

// What an adapter holds, read from the safetensors headers only
struct AdapterInfo {
    bool                     ok = false;
    std::string              error;
    int                      lm_keys  = 0;
    int                      dit_keys = 0;
    int                      ignored  = 0;  // keys naming nothing in the backbone
    std::string              first_ignored;
    std::string              trigger;       // __metadata__ trigger when a trainer wrote one
    std::vector<std::string> files;
};

static AdapterInfo adapter_inspect(const std::string & path) {
    AdapterInfo info;
    info.files = adapter_files(path);
    if (info.files.empty()) {
        info.error = "no .safetensors in " + path;
        return info;
    }
    for (const auto & file : info.files) {
        STFile st = {};
        if (!st_open(&st, file.c_str())) {
            info.error = "cannot read " + file;
            return info;
        }
        auto meta = adapter_read_metadata(st);
        if (info.trigger.empty() && meta.count("trigger")) {
            info.trigger = meta["trigger"];
        }
        std::string refused = adapter_unsupported(st, meta);
        if (!refused.empty()) {
            info.error = file + ": " + refused;
            st_close(&st);
            return info;
        }
        for (const auto & e : st.entries) {
            AdapterKey k;
            // HOT-Step's trainer marks its files with hot_step.* tensors of its own
            if (e.name.rfind("hot_step.", 0) == 0) {
                continue;
            }
            if (!adapter_parse_key(e.name, &k)) {
                if (info.ignored++ == 0) {
                    info.first_ignored = e.name;
                }
                continue;
            }
            (k.half == ADAPTER_LM ? info.lm_keys : info.dit_keys)++;
        }
        st_close(&st);
    }
    if (info.lm_keys + info.dit_keys == 0) {
        info.error = "no key of " + path + " names a MiniMax Music 3 tensor";
        return info;
    }
    info.ok = true;
    return info;
}

// Everything one adapter file says about one fused or plain module
struct AdapterModule {
    const STEntry * a     = nullptr;
    const STEntry * b     = nullptr;
    const STEntry * alpha = nullptr;
    const STEntry * w1    = nullptr;
    const STEntry * w1a   = nullptr;
    const STEntry * w1b   = nullptr;
    const STEntry * w2    = nullptr;
    const STEntry * w2a   = nullptr;
    const STEntry * w2b   = nullptr;
    const STEntry * diff  = nullptr;
    const STEntry * full  = nullptr;
    bool            fused = false;
};

// One contribution to one GGUF tensor, its factors already on the host in F32
struct AdapterTerm {
    AdapterRole        kind;                            // ROLE_LORA_A for LoRA, ROLE_LOKR_W1, ROLE_DIFF, ROLE_FULL
    float              scale;                           // user scale times alpha over rank
    std::vector<float> f0;                              // LoRA A [rank, in], LoKr w1 slice [l, b], diff or full
    std::vector<float> f1;                              // LoRA B rows [rows, rank], LoKr W2 [c, d]
    int64_t            rank = 0;
    int64_t            l = 0, bcols = 0, c = 0, d = 0;  // LoKr kron dims
};

// Row ranges of the projections a fused site splits into, from the GGUF shapes
static bool adapter_fused_parts(const GGUFModel &                              gf,
                                const std::string &                            module,
                                std::vector<std::pair<std::string, int64_t>> * parts) {
    std::string              head;
    std::vector<std::string> names;
    if (adapter_ends_with(module, "to_qkv", &head)) {
        names = { head + "to_q", head + "to_k", head + "to_v" };
    } else if (adapter_ends_with(module, "qkv_proj", &head)) {
        names = { head + "q_proj", head + "k_proj", head + "v_proj" };
    } else if (adapter_ends_with(module, "gate_up_proj", &head)) {
        names = { head + "gate_proj", head + "up_proj" };
    } else {
        return false;
    }
    for (const auto & n : names) {
        struct ggml_tensor * t = gf_meta(gf, (n + ".weight").c_str());
        if (!t) {
            return false;
        }
        parts->push_back({ n, t->ne[1] });
    }
    return true;
}

// Reads one adapter file into per tensor terms for one half. Returns false
// with a reason on anything that cannot merge exactly: a shape that does not
// match the backbone, a dtype we cannot read, a factor without its pair.
static bool adapter_collect(const std::string &                               file,
                            const AdapterConfig &                             config,
                            AdapterHalf                                       half,
                            float                                             user_scale,
                            const GGUFModel &                                 gf,
                            std::map<std::string, std::vector<AdapterTerm>> * terms,
                            std::string *                                     error) {
    STFile st = {};
    if (!st_open(&st, file.c_str())) {
        *error = "cannot read " + file;
        return false;
    }
    auto  meta       = adapter_read_metadata(st);
    float meta_alpha = meta.count("alpha") ? (float) atof(meta["alpha"].c_str()) : 0.0f;
    float lokr_dim   = meta.count("lokr_dim") ? (float) atof(meta["lokr_dim"].c_str()) : 0.0f;
    // SimpleTuner writes the flag bare or namespaced by its component
    bool  gate_first = false;
    for (const char * k : { "swiglu_gate_first", "transformer.swiglu_gate_first" }) {
        auto it = meta.find(k);
        if (it != meta.end()) {
            std::string v = it->second;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char) tolower(c); });
            gate_first = v == "true" || v == "1";
            break;
        }
    }

    // group the keys of this half by module and target
    std::map<std::string, AdapterModule> modules;
    for (const auto & e : st.entries) {
        AdapterKey k;
        if (!adapter_parse_key(e.name, &k) || k.half != half) {
            continue;
        }
        std::string     id = k.module + (k.bias ? ".bias" : ".weight");
        AdapterModule & m  = modules[id];
        m.fused            = k.fused;
        switch (k.role) {
            case ROLE_LORA_A:
                m.a = &e;
                break;
            case ROLE_LORA_B:
                m.b = &e;
                break;
            case ROLE_ALPHA:
                m.alpha = &e;
                break;
            case ROLE_LOKR_W1:
                m.w1 = &e;
                break;
            case ROLE_LOKR_W1A:
                m.w1a = &e;
                break;
            case ROLE_LOKR_W1B:
                m.w1b = &e;
                break;
            case ROLE_LOKR_W2:
                m.w2 = &e;
                break;
            case ROLE_LOKR_W2A:
                m.w2a = &e;
                break;
            case ROLE_LOKR_W2B:
                m.w2b = &e;
                break;
            case ROLE_DIFF:
                m.diff = &e;
                break;
            case ROLE_FULL:
                m.full = &e;
                break;
        }
    }

    auto fail = [&](const std::string & why) {
        *error = file + ": " + why;
        st_close(&st);
        return false;
    };
    auto read = [&](const STEntry * e, std::vector<float> * out) {
        out->resize((size_t) adapter_numel(*e));
        return adapter_to_f32(st_data(st, *e), out->data(), (int64_t) out->size(), e->dtype);
    };

    for (const auto & kv : modules) {
        const std::string &   id = kv.first;
        const AdapterModule & m  = kv.second;
        std::string           module;
        bool                  is_bias = adapter_ends_with(id, ".bias", &module);
        if (!is_bias) {
            adapter_ends_with(id, ".weight", &module);
        }

        // targets: the tensor itself, or the projections a fused site splits into
        std::vector<std::pair<std::string, int64_t>> parts;
        if (m.fused) {
            if (!adapter_fused_parts(gf, module, &parts)) {
                return fail("no projections behind " + module);
            }
        } else {
            struct ggml_tensor * t = gf_meta(gf, id.c_str());
            if (!t) {
                return fail("the model has no tensor " + id);
            }
            parts.push_back({ module, t->ne[1] });
        }
        int64_t total_rows = 0;
        for (const auto & p : parts) {
            total_rows += p.second;
        }

        float alpha = 0.0f;
        if (m.alpha) {
            float v = 0.0f;
            if (adapter_numel(*m.alpha) != 1 || !adapter_to_f32(st_data(st, *m.alpha), &v, 1, m.alpha->dtype)) {
                return fail("unreadable alpha of " + module);
            }
            alpha = v;
        } else if (meta_alpha > 0.0f) {
            alpha = meta_alpha;
        } else if (config.alpha > 0.0f) {
            alpha = config.alpha;
        }

        if (m.a || m.b) {
            if (!m.a || !m.b) {
                return fail("LoRA factor without its pair on " + module);
            }
            if (m.a->n_dims != 2 || m.b->n_dims != 2 || m.a->shape[0] != m.b->shape[1]) {
                return fail("LoRA factor shapes disagree on " + module);
            }
            int64_t rank = m.a->shape[0];
            int64_t in   = m.a->shape[1];
            if (m.b->shape[0] != total_rows) {
                return fail("LoRA output rows " + std::to_string(m.b->shape[0]) + " != " + std::to_string(total_rows) +
                            " on " + module);
            }
            std::vector<float> a, b;
            if (!read(m.a, &a) || !read(m.b, &b)) {
                return fail("unsupported LoRA dtype on " + module);
            }
            std::string block;
            if (gate_first && adapter_ends_with(module, ".ff_in", &block)) {
                // gate | value rows to the value | gate order of the GGUF
                if (total_rows % 2 != 0) {
                    return fail("odd ff_in height on " + module);
                }
                int64_t            mid = total_rows / 2;
                std::vector<float> swapped(b.size());
                std::copy(b.begin() + mid * rank, b.end(), swapped.begin());
                std::copy(b.begin(), b.begin() + mid * rank, swapped.begin() + mid * rank);
                b.swap(swapped);
            }
            float   divisor = config.rslora ? std::sqrt((float) rank) : (float) rank;
            float   scaling = user_scale * (alpha > 0.0f ? alpha / divisor : 1.0f);
            int64_t row0    = 0;
            for (const auto & p : parts) {
                struct ggml_tensor * t = gf_meta(gf, (p.first + ".weight").c_str());
                if (t->ne[0] != in) {
                    return fail("LoRA input width " + std::to_string(in) + " != " + std::to_string(t->ne[0]) + " on " +
                                p.first);
                }
                AdapterTerm term;
                term.kind  = ROLE_LORA_A;
                term.scale = scaling;
                term.rank  = rank;
                term.f0    = a;
                term.f1.assign(b.begin() + row0 * rank, b.begin() + (row0 + p.second) * rank);
                (*terms)[p.first + ".weight"].push_back(std::move(term));
                row0 += p.second;
            }
        }

        std::string ff_block;
        if (m.w1 && gate_first && adapter_ends_with(module, ".ff_in", &ff_block)) {
            return fail("a gate first LoKr on ff_in cannot be merged");
        }
        bool has_w1 = m.w1 || m.w1a || m.w1b;
        if (!has_w1 && (m.w2 || m.w2a || m.w2b)) {
            return fail("LoKr w2 without its w1 on " + module);
        }
        if (has_w1) {
            bool factor    = m.w2a && m.w2b;
            bool factor_w1 = m.w1a && m.w1b;
            if (factor == (m.w2 != nullptr) || factor_w1 == (m.w1 != nullptr) || (m.w1 && m.w1->n_dims != 2)) {
                return fail("incomplete LoKr module " + module);
            }
            std::vector<float> w1, w2;
            int64_t            c, d, r = 0;
            if (factor) {
                std::vector<float> w2a, w2b;
                if (!read(m.w2a, &w2a) || !read(m.w2b, &w2b)) {
                    return fail("unsupported LoKr dtype on " + module);
                }
                c = m.w2a->shape[0];
                r = m.w2a->shape[1];
                d = m.w2b->shape[1];
                if (m.w2b->shape[0] != r) {
                    return fail("LoKr rank mismatch on " + module);
                }
                // W2 = w2_a @ w2_b on the host, it is small
                w2.assign((size_t) (c * d), 0.0f);
                for (int64_t i = 0; i < c; i++) {
                    for (int64_t k = 0; k < r; k++) {
                        float av = w2a[(size_t) (i * r + k)];
                        for (int64_t j = 0; j < d; j++) {
                            w2[(size_t) (i * d + j)] += av * w2b[(size_t) (k * d + j)];
                        }
                    }
                }
            } else {
                if (!read(m.w2, &w2)) {
                    return fail("unsupported LoKr dtype on " + module);
                }
                c = m.w2->shape[0];
                d = m.w2->shape[1];
            }
            int64_t a_rows, b_cols;
            if (factor_w1) {
                // W1 = w1_a @ w1_b, the same way
                std::vector<float> w1a, w1b;
                if (!read(m.w1a, &w1a) || !read(m.w1b, &w1b)) {
                    return fail("unsupported LoKr dtype on " + module);
                }
                int64_t r1 = m.w1a->shape[1];
                a_rows     = m.w1a->shape[0];
                b_cols     = m.w1b->shape[1];
                if (m.w1b->shape[0] != r1) {
                    return fail("LoKr w1 rank mismatch on " + module);
                }
                if (r == 0) {
                    r = r1;
                }
                w1.assign((size_t) (a_rows * b_cols), 0.0f);
                for (int64_t i = 0; i < a_rows; i++) {
                    for (int64_t k = 0; k < r1; k++) {
                        float av = w1a[(size_t) (i * r1 + k)];
                        for (int64_t j = 0; j < b_cols; j++) {
                            w1[(size_t) (i * b_cols + j)] += av * w1b[(size_t) (k * b_cols + j)];
                        }
                    }
                }
            } else {
                if (!read(m.w1, &w1)) {
                    return fail("unsupported LoKr dtype on " + module);
                }
                a_rows = m.w1->shape[0];
                b_cols = m.w1->shape[1];
            }
            // the rows this file holds for the module: a native split file
            // carries the w1 slice of its own projection already
            if (a_rows * c != total_rows) {
                return fail("LoKr rows " + std::to_string(a_rows * c) + " != " + std::to_string(total_rows) + " on " +
                            module);
            }
            float dim = lokr_dim > 0.0f ? lokr_dim : (float) r;
            float scaling =
                user_scale * ((!(factor || factor_w1) || alpha <= 0.0f || dim <= 0.0f) ? 1.0f : alpha / dim);
            int64_t row0 = 0;
            for (const auto & p : parts) {
                struct ggml_tensor * t = gf_meta(gf, (p.first + ".weight").c_str());
                if (row0 % c != 0 || p.second % c != 0 || t->ne[0] != b_cols * d) {
                    return fail("LoKr blocks do not tile " + p.first);
                }
                AdapterTerm term;
                term.kind  = ROLE_LOKR_W1;
                term.scale = scaling;
                term.l     = p.second / c;
                term.bcols = b_cols;
                term.c     = c;
                term.d     = d;
                term.f0.assign(w1.begin() + (row0 / c) * b_cols, w1.begin() + (row0 / c + term.l) * b_cols);
                term.f1 = w2;
                (*terms)[p.first + ".weight"].push_back(std::move(term));
                row0 += p.second;
            }
        }

        for (const STEntry * e : { m.diff, m.full }) {
            if (!e) {
                continue;
            }
            struct ggml_tensor * t = gf_meta(gf, id.c_str());
            if (!t || adapter_numel(*e) != ggml_nelements(t)) {
                return fail("whole tensor of " + id + " does not match the model shape");
            }
            AdapterTerm term;
            term.kind  = e == m.diff ? ROLE_DIFF : ROLE_FULL;
            term.scale = user_scale;
            if (!read(e, &term.f0)) {
                return fail("unsupported dtype on " + id);
            }
            (*terms)[id].push_back(std::move(term));
        }
    }
    st_close(&st);
    return true;
}

// True when the backend encodes F32 into this type itself
static bool adapter_backend_can_encode(ggml_backend_t backend, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }
    size_t                  meta   = ggml_tensor_overhead() * 4 + 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    struct ggml_tensor *    src    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    struct ggml_tensor *    dst    = ggml_cast(ctx, src, type);
    bool                    ok     = ggml_backend_supports_op(backend, dst);
    ggml_free(ctx);
    return ok;
}

// Whether the backend can decode a tensor type to F32 in the merge graph; the
// CUDA copy kernels have no path from the K quants to F32
static bool adapter_backend_can_decode(ggml_backend_t backend, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }
    size_t                  meta   = ggml_tensor_overhead() * 4 + 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    struct ggml_tensor *    src    = ggml_new_tensor_1d(ctx, type, ggml_blck_size(type));
    struct ggml_tensor *    dst    = ggml_cast(ctx, src, GGML_TYPE_F32);
    bool                    ok     = ggml_backend_supports_op(backend, dst);
    ggml_free(ctx);
    return ok;
}

// F32 back to the tensor type on the host, for the types the backend cannot encode
static size_t adapter_requant(const float * src, void * dst, int64_t nel, int64_t n_per_row, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) nel * sizeof(float));
        return (size_t) nel * sizeof(float);
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits->is_quantized) {
        int64_t nrows = nel / n_per_row;
        ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);
        return ggml_row_size(type, n_per_row) * (size_t) nrows;
    }
    if (traits->from_float_ref) {
        traits->from_float_ref(src, dst, nel);
        return (size_t) nel * traits->type_size;
    }
    return 0;
}

// Sums every term of one tensor into its staged bytes in one backend graph
static bool adapter_merge_tensor(WeightCtx *                      wctx,
                                 WeightCtx::PendingCopy *         pc,
                                 int64_t                          ne0,
                                 int64_t                          ne1,
                                 const std::vector<AdapterTerm> & terms,
                                 ggml_backend_t                   backend) {
    enum ggml_type ttype   = pc->tensor->type;
    size_t         base_nb = ggml_row_size(ttype, ne0) * (size_t) ne1;
    if (base_nb != pc->nbytes) {
        fprintf(stderr, "[Adapter] ERROR: staged size of %s disagrees with its shape\n", ggml_get_name(pc->tensor));
        return false;
    }
    bool                            encode_ok = adapter_backend_can_encode(backend, ttype);
    bool                            decode_ok = adapter_backend_can_decode(backend, ttype);
    const struct ggml_type_traits * traits    = ggml_get_type_traits(ttype);
    if (!decode_ok && !traits->to_float) {
        fprintf(stderr, "[Adapter] ERROR: cannot decode %s (%s) to merge into it\n", ggml_get_name(pc->tensor),
                ggml_type_name(ttype));
        return false;
    }

    size_t meta = ggml_tensor_overhead() * (16 + 16 * terms.size()) + ggml_graph_overhead() + 64 * 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }

    std::vector<std::pair<struct ggml_tensor *, const std::vector<float> *>> uploads;
    // a type the backend cannot decode is decoded here and goes up as F32
    std::vector<float>                                                       host_base;
    struct ggml_tensor *                                                     tbase = nullptr;
    struct ggml_tensor *                                                     tbf   = nullptr;
    if (decode_ok) {
        tbase = ggml_new_tensor_2d(ctx, ttype, ne0, ne1);
        tbf   = ttype == GGML_TYPE_F32 ? tbase : ggml_cast(ctx, tbase, GGML_TYPE_F32);
    } else {
        host_base.resize((size_t) (ne0 * ne1));
        traits->to_float(pc->src, host_base.data(), ne0 * ne1);
        tbf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        uploads.push_back({ tbf, &host_base });
    }
    struct ggml_tensor * acc = tbf;

    for (const auto & t : terms) {
        struct ggml_tensor * delta = nullptr;
        if (t.kind == ROLE_LORA_A) {
            // A row major [rank, in] is ne (in, rank), B rows [rows, rank] is ne (rank, rows).
            // PEFT merges in the adapter dtype, BF16, so the factors and the delta round there.
            struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, t.rank);
            struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t.rank, ne1);
            uploads.push_back({ ta, &t.f0 });
            uploads.push_back({ tb, &t.f1 });
            struct ggml_tensor * ta_r = ggml_cast(ctx, ggml_cast(ctx, ta, GGML_TYPE_BF16), GGML_TYPE_F32);
            struct ggml_tensor * tb_r = ggml_cast(ctx, ggml_cast(ctx, tb, GGML_TYPE_BF16), GGML_TYPE_F32);
            struct ggml_tensor * ta_t = ggml_cont(ctx, ggml_transpose(ctx, ta_r));
            delta                     = ggml_mul_mat(ctx, ta_t, tb_r);
            delta                     = ggml_cast(ctx, ggml_cast(ctx, delta, GGML_TYPE_BF16), GGML_TYPE_F32);
            delta                     = ggml_scale(ctx, delta, t.scale);
        } else if (t.kind == ROLE_LOKR_W1) {
            // kron(w1 [l, b], W2 [c, d]) = [l*c, b*d], the layout of the LyCORIS path in acestep.cpp
            int64_t              a = t.l, b = t.bcols, c = t.c, d = t.d;
            struct ggml_tensor * tw1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b, a);
            struct ggml_tensor * tw2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, c);
            uploads.push_back({ tw1, &t.f0 });
            uploads.push_back({ tw2, &t.f1 });
            struct ggml_tensor * w1s = ggml_scale(ctx, tw1, t.scale);
            struct ggml_tensor * outer =
                ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w1s, 1, a * b), ggml_reshape_2d(ctx, tw2, 1, c * d));
            struct ggml_tensor * k4 = ggml_reshape_4d(ctx, outer, b, a, d, c);
            struct ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k4, 1, 3, 0, 2));
            delta                   = ggml_reshape_2d(ctx, kp, b * d, a * c);
            delta                   = ggml_cast(ctx, ggml_cast(ctx, delta, GGML_TYPE_BF16), GGML_TYPE_F32);
        } else {
            struct ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
            uploads.push_back({ tw, &t.f0 });
            // diff adds as is, a full weight adds its distance from the base
            delta = t.kind == ROLE_DIFF ? tw : ggml_sub(ctx, tw, tbf);
            delta = ggml_scale(ctx, delta, t.scale);
        }
        acc = ggml_add(ctx, acc, delta);
    }

    struct ggml_tensor * tout  = encode_ok && ttype != GGML_TYPE_F32 ? ggml_cast(ctx, acc, ttype) : acc;
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tout);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    if (tbase) {
        ggml_backend_tensor_set(tbase, pc->src, 0, base_nb);
    }
    for (const auto & u : uploads) {
        ggml_backend_tensor_set(u.first, u.second->data(), 0, u.second->size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Adapter] ERROR: merging into %s failed on the backend\n", ggml_get_name(pc->tensor));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // the staged bytes are the tensor's own size; the F32 result has its own buffer
    std::unique_ptr<float[]> staging(new float[base_nb / sizeof(float) + 1]);
    bool                     ok = true;
    if (encode_ok || ttype == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tout, staging.get(), 0, base_nb);
    } else {
        std::vector<float> merged((size_t) (ne0 * ne1));
        ggml_backend_tensor_get(tout, merged.data(), 0, merged.size() * sizeof(float));
        ok = adapter_requant(merged.data(), staging.get(), ne0 * ne1, ne0, ttype) == base_nb;
    }
    if (ok) {
        pc->src = staging.get();
        wctx->staging.push_back(std::move(staging));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Merges one half's adapter list into the staged weights. Call after every
// GGUF load of the half and before wctx_alloc. A failure leaves the half
// unusable: the caller refuses the load rather than serve a partly adapted
// model.
static bool adapter_apply(WeightCtx *                      wctx,
                          const GGUFModel &                gf,
                          AdapterHalf                      half,
                          const std::vector<AdapterSpec> & specs,
                          ggml_backend_t                   backend) {
    if (specs.empty()) {
        return true;
    }
    const char * tag = half == ADAPTER_LM ? "LM" : "DiT";

    std::map<std::string, std::vector<AdapterTerm>> terms;
    for (const auto & spec : specs) {
        std::vector<std::string> files = adapter_files(spec.path);
        if (files.empty()) {
            fprintf(stderr, "[Adapter] ERROR: no .safetensors in %s\n", spec.path.c_str());
            return false;
        }
        // adapter_config.json belongs to an adapter directory; a lone file
        // beside others carries its settings in its own keys and metadata
        AdapterConfig config = adapter_is_dir(spec.path) ? adapter_read_config(spec.path) : AdapterConfig{};
        if (!config.refused.empty()) {
            fprintf(stderr, "[Adapter] ERROR: %s: %s\n", spec.path.c_str(), config.refused.c_str());
            return false;
        }
        AdapterInfo info = adapter_inspect(spec.path);
        if (info.ignored > 0) {
            fprintf(stderr, "[Adapter] WARNING: %d keys of %s name nothing in the model, the first %s\n", info.ignored,
                    spec.path.c_str(), info.first_ignored.c_str());
        }

        // two files of one adapter touching one tensor would apply it twice
        std::map<std::string, std::vector<AdapterTerm>> own;
        for (const auto & file : files) {
            std::map<std::string, std::vector<AdapterTerm>> part;
            std::string                                     error;
            if (!adapter_collect(file, config, half, spec.scale, gf, &part, &error)) {
                fprintf(stderr, "[Adapter] ERROR: %s\n", error.c_str());
                return false;
            }
            for (auto & kv : part) {
                if (own.count(kv.first)) {
                    fprintf(stderr, "[Adapter] ERROR: two files of %s both adapt %s\n", spec.path.c_str(),
                            kv.first.c_str());
                    return false;
                }
                own[kv.first] = std::move(kv.second);
            }
        }
        if (own.empty()) {
            fprintf(stderr, "[Adapter] ERROR: %s changes nothing in the %s\n", spec.path.c_str(), tag);
            return false;
        }
        for (auto & kv : own) {
            auto & dst = terms[kv.first];
            for (auto & t : kv.second) {
                dst.push_back(std::move(t));
            }
        }
        fprintf(stderr, "[Adapter] %s %s: %zu tensors at scale %.2f\n", tag, spec.path.c_str(), own.size(),
                (double) spec.scale);
    }

    // staged copy of every GGUF tensor: plain loads by name, fused parts by
    // the mapped bytes they were staged from
    std::unordered_map<std::string, size_t>  by_name;
    std::unordered_map<const void *, size_t> by_src;
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        const auto & pc = wctx->pending[i];
        by_src[pc.src]  = i;
        const char * n  = ggml_get_name(pc.tensor);
        if (n && n[0] && pc.offset == 0 && pc.nbytes == ggml_nbytes(pc.tensor)) {
            by_name[n] = i;
        }
    }

    int merged = 0;
    for (const auto & kv : terms) {
        const std::string &  name = kv.first;
        struct ggml_tensor * meta = gf_meta(gf, name.c_str());
        if (!meta) {
            fprintf(stderr, "[Adapter] ERROR: the model has no tensor %s\n", name.c_str());
            return false;
        }
        size_t idx;
        auto   it = by_name.find(name);
        if (it != by_name.end()) {
            idx = it->second;
        } else {
            auto src = by_src.find(gf_get_data(gf, name.c_str()));
            if (src == by_src.end()) {
                fprintf(stderr, "[Adapter] ERROR: %s is not a weight this module loads\n", name.c_str());
                return false;
            }
            idx = src->second;
        }
        int64_t ne0 = meta->ne[0];
        int64_t ne1 = ggml_n_dims(meta) > 1 ? meta->ne[1] : 1;
        if (!adapter_merge_tensor(wctx, &wctx->pending[idx], ne0, ne1, kv.second, backend)) {
            fprintf(stderr, "[Adapter] ERROR: merge failed on %s\n", name.c_str());
            return false;
        }
        merged++;
    }
    fprintf(stderr, "[Adapter] %s merged %d tensors from %zu adapters\n", tag, merged, specs.size());
    return true;
}

// One adapter available to requests: a .safetensors file, or a directory of
// them, directly under the adapter directory
struct AdapterEntry {
    std::string name;  // the file or directory name, what a request names
    std::string path;
    AdapterInfo info;
};

static std::vector<AdapterEntry> adapter_scan(const std::string & dir) {
    std::vector<AdapterEntry> out;
    if (dir.empty()) {
        return out;
    }
    for (const auto & name : adapter_list_dir(dir, false)) {
        if (adapter_is_safetensors(name)) {
            out.push_back({ name, dir + "/" + name, {} });
        }
    }
    for (const auto & name : adapter_list_dir(dir, true)) {
        if (!adapter_files(dir + "/" + name).empty()) {
            out.push_back({ name, dir + "/" + name, {} });
        }
    }
    for (auto & e : out) {
        e.info = adapter_inspect(e.path);
    }
    return out;
}

// Resolves a request name against the adapter directory. Names are plain
// entries of the directory, never paths.
static bool adapter_resolve(const std::string & dir, const std::string & name, std::string * path) {
    if (dir.empty() || name.empty() || name == "." || name == ".." || name.find_first_of("/\\:") != std::string::npos) {
        return false;
    }
    std::string p = dir + "/" + name;
    if (adapter_is_dir(p) ? adapter_files(p).empty() : !(adapter_is_file(p) && adapter_is_safetensors(name))) {
        return false;
    }
    *path = p;
    return true;
}
