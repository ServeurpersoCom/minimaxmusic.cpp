#pragma once
// model-registry.h: scan a directory for GGUF models.
//
// Reads only GGUF headers (no weight data) to classify each file by its
// general.architecture KV into the five MM3 component buckets.
//
// Usage:
//   ModelRegistry reg;
//   registry_scan(&reg, "./models");
//   const ModelEntry * dit = registry_find(reg.dit, "MiniMax-Music3-transformer-F32.gguf");

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <sys/stat.h>
#endif

struct ModelEntry {
    std::string name;  // filename (e.g. "MiniMax-Music3-language_model-BF16.gguf")
    std::string path;  // full path
};

struct ModelRegistry {
    std::vector<ModelEntry> lm;
    std::vector<ModelEntry> depth;
    std::vector<ModelEntry> cond;
    std::vector<ModelEntry> dit;
    std::vector<ModelEntry> vae;
};

// find an entry by name in a bucket. returns NULL if not found.
static const ModelEntry * registry_find(const std::vector<ModelEntry> & bucket, const char * name) {
    for (const auto & e : bucket) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// resolve a requested model name against a bucket. An empty request keeps
// the loaded path, or falls to the first entry. Unknown names and empty
// buckets log and resolve to an empty path.
static std::string registry_resolve(const std::vector<ModelEntry> & bucket,
                                    const std::string &             requested,
                                    const char *                    component,
                                    const std::string &             loaded = "") {
    if (!requested.empty()) {
        const ModelEntry * e = registry_find(bucket, requested.c_str());
        if (!e) {
            fprintf(stderr, "[Registry] ERROR: unknown %s model %s\n", component, requested.c_str());
            return "";
        }
        return e->path;
    }
    if (!loaded.empty()) {
        return loaded;
    }
    if (bucket.empty()) {
        fprintf(stderr, "[Registry] ERROR: no %s model found\n", component);
        return "";
    }
    return bucket.front().path;
}

// classify a GGUF file by reading its header.
// returns the mm3-* architecture string, or "" if unrecognized.
static std::string registry_classify_gguf(const char * path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context *   ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return "";
    }
    std::string arch;
    int64_t     idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        arch = gguf_get_val_str(ctx, idx);
    }
    gguf_free(ctx);
    if (arch == "mm3-lm" || arch == "mm3-depth" || arch == "mm3-cond" || arch == "mm3-dit" || arch == "mm3-vae") {
        return arch;
    }
    return "";
}

// check if a string ends with a suffix
static bool str_ends_with(const std::string & s, const char * suffix) {
    size_t slen = strlen(suffix);
    return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

#ifdef _WIN32

// scan a directory for files matching a pattern (Windows)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

#else

// scan a directory for files (POSIX)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        // skip directories
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

#endif

// path separator
#ifdef _WIN32
#    define REGISTRY_SEP "\\"
#else
#    define REGISTRY_SEP "/"
#endif

// scan a directory for .gguf files, classify each by architecture.
// returns true if at least one model was found.
static bool registry_scan(ModelRegistry * reg, const char * models_dir) {
    std::vector<std::string> files;
    registry_list_dir(models_dir, &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".gguf")) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        std::string type = registry_classify_gguf(full.c_str());
        if (type.empty()) {
            fprintf(stderr, "[Registry] WARNING: skipping %s (unknown architecture)\n", fname.c_str());
            continue;
        }

        ModelEntry entry = { fname, full };
        if (type == "mm3-lm") {
            reg->lm.push_back(entry);
        } else if (type == "mm3-depth") {
            reg->depth.push_back(entry);
        } else if (type == "mm3-cond") {
            reg->cond.push_back(entry);
        } else if (type == "mm3-dit") {
            reg->dit.push_back(entry);
        } else if (type == "mm3-vae") {
            reg->vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s -> %s\n", fname.c_str(), type.c_str());
        count++;
    }

    return count > 0;
}
