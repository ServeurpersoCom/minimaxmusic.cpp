#pragma once
// debug.h: tensor dump utilities for the Python cossim harnesses
// Dumps raw f32 arrays to binary files; the comparison side lives in tests/.
// File format: [int32 ndims] [int32 dim0] [int32 dim1] ... [float data...]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct DebugDumper {
    char dir[512];
    bool enabled;
};

static void debug_init(DebugDumper * d, const char * dir) {
    d->enabled = (dir != nullptr);
    if (d->enabled) {
        snprintf(d->dir, sizeof(d->dir), "%s", dir);
    }
}

// Dump f32 tensor to binary file
// Format: [ndims:i32] [shape:i32 x ndims] [data:f32 x numel]
static void debug_dump(const DebugDumper * d, const char * name, const float * data, const int * shape, int ndims) {
    if (!d->enabled) {
        return;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", d->dir, name);

    int numel = 1;
    for (int i = 0; i < ndims; i++) {
        numel *= shape[i];
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Debug] Cannot write %s\n", path);
        return;
    }

    fwrite(&ndims, sizeof(int32_t), 1, f);
    fwrite(shape, sizeof(int32_t), ndims, f);
    fwrite(data, sizeof(float), numel, f);
    fclose(f);

    // Print first 4 values for quick sanity check
    fprintf(stderr, "[Debug] %s: [", name);
    for (int i = 0; i < ndims; i++) {
        fprintf(stderr, "%s%d", i ? ", " : "", shape[i]);
    }
    fprintf(stderr, "] first4:");
    for (int i = 0; i < 4 && i < numel; i++) {
        fprintf(stderr, " %.6f", data[i]);
    }
    fprintf(stderr, "\n");
}

// Convenience: dump 2D tensor [rows, cols]
static void debug_dump_2d(const DebugDumper * d, const char * name, const float * data, int dim0, int dim1) {
    int shape[2] = { dim0, dim1 };
    debug_dump(d, name, data, shape, 2);
}

// Convenience: dump 1D tensor [n]
static void debug_dump_1d(const DebugDumper * d, const char * name, const float * data, int n) {
    debug_dump(d, name, data, &n, 1);
}
