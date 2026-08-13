// request.cpp: MM3 request JSON read/write (yyjson)

#include "request.h"

#include "task-types.h"
#include "yyjson.h"

#include <random>
#include <string>

static const yyjson_write_flag WRITE_FLAGS =
    YYJSON_WRITE_PRETTY | YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_FP_TO_FIXED(2);

void request_init(MM3Request * r) {
    r->caption = "";
    r->lyrics  = "";

    r->duration = 60.0f;
    r->steps    = 30;
    r->seed     = -1;
    r->lm_seed  = -1;

    r->lm_cfg   = 1.5f;
    r->lm_top_k = 50;
    r->dit_cfg  = 1.7f;

    r->peak_clip     = 10;
    r->output_format = OUTPUT_FORMAT_MP3;
    r->mp3_bitrate   = 128;

    r->lm_model    = "";
    r->depth_model = "";
    r->cond_model  = "";
    r->dit_model   = "";
    r->vae_model   = "";
}

// helper: get yyjson string as std::string
static inline std::string yy_str(yyjson_val * v) {
    return std::string(yyjson_get_str(v), yyjson_get_len(v));
}

// populate MM3Request fields from a yyjson object (must be pre-initialized)
static void request_parse_obj(yyjson_val * obj, MM3Request * r) {
    yyjson_val * v;

    if ((v = yyjson_obj_get(obj, "caption")) && yyjson_is_str(v)) {
        r->caption = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lyrics")) && yyjson_is_str(v)) {
        r->lyrics = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "output_format")) && yyjson_is_str(v)) {
        r->output_format = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_model")) && yyjson_is_str(v)) {
        r->lm_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "depth_model")) && yyjson_is_str(v)) {
        r->depth_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "cond_model")) && yyjson_is_str(v)) {
        r->cond_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "dit_model")) && yyjson_is_str(v)) {
        r->dit_model = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "vae_model")) && yyjson_is_str(v)) {
        r->vae_model = yy_str(v);
    }

    if ((v = yyjson_obj_get(obj, "duration")) && yyjson_is_num(v)) {
        r->duration = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "steps")) && yyjson_is_int(v)) {
        r->steps = (int) yyjson_get_sint(v);
    }
    if ((v = yyjson_obj_get(obj, "seed")) && yyjson_is_int(v)) {
        r->seed = yyjson_get_sint(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_seed")) && yyjson_is_int(v)) {
        r->lm_seed = yyjson_get_sint(v);
    }

    if ((v = yyjson_obj_get(obj, "lm_cfg")) && yyjson_is_num(v)) {
        r->lm_cfg = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_top_k")) && yyjson_is_int(v)) {
        r->lm_top_k = (int) yyjson_get_sint(v);
    }
    if ((v = yyjson_obj_get(obj, "dit_cfg")) && yyjson_is_num(v)) {
        r->dit_cfg = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "peak_clip")) && yyjson_is_int(v)) {
        r->peak_clip = (int) yyjson_get_sint(v);
    }
    if ((v = yyjson_obj_get(obj, "mp3_bitrate")) && yyjson_is_int(v)) {
        r->mp3_bitrate = (int) yyjson_get_sint(v);
    }
}

bool request_parse_json(MM3Request * r, const char * json) {
    yyjson_doc * doc = yyjson_read(json, strlen(json), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }
    request_parse_obj(root, r);
    yyjson_doc_free(doc);
    return true;
}

std::string request_to_json(const MM3Request * r, bool sparse) {
    MM3Request def;
    request_init(&def);

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    auto put_str = [&](const char * key, const std::string & v, const std::string & d) {
        if (!sparse || v != d) {
            yyjson_mut_obj_add_strncpy(doc, root, key, v.c_str(), v.size());
        }
    };
    auto put_f = [&](const char * key, float v, float d) {
        if (!sparse || v != d) {
            yyjson_mut_obj_add_real(doc, root, key, v);
        }
    };
    auto put_i = [&](const char * key, int64_t v, int64_t d) {
        if (!sparse || v != d) {
            yyjson_mut_obj_add_sint(doc, root, key, v);
        }
    };

    put_str("caption", r->caption, def.caption);
    put_str("lyrics", r->lyrics, def.lyrics);
    put_f("duration", r->duration, def.duration);
    put_i("steps", r->steps, def.steps);
    put_i("seed", r->seed, def.seed);
    put_i("lm_seed", r->lm_seed, def.lm_seed);
    put_f("lm_cfg", r->lm_cfg, def.lm_cfg);
    put_i("lm_top_k", r->lm_top_k, def.lm_top_k);
    put_f("dit_cfg", r->dit_cfg, def.dit_cfg);
    put_i("peak_clip", r->peak_clip, def.peak_clip);
    put_str("output_format", r->output_format, def.output_format);
    put_i("mp3_bitrate", r->mp3_bitrate, def.mp3_bitrate);
    put_str("lm_model", r->lm_model, def.lm_model);
    put_str("depth_model", r->depth_model, def.depth_model);
    put_str("cond_model", r->cond_model, def.cond_model);
    put_str("dit_model", r->dit_model, def.dit_model);
    put_str("vae_model", r->vae_model, def.vae_model);

    char *      s   = yyjson_mut_write(doc, WRITE_FLAGS, NULL);
    std::string out = s ? s : "{}";
    free(s);
    yyjson_mut_doc_free(doc);
    return out;
}

// hardware random value in [0, UINT32_MAX], positive in int64_t
static int64_t random_seed() {
    std::random_device rd;
    return (int64_t) (uint32_t) rd();
}

void request_resolve_seed(MM3Request * r) {
    if (r->seed < 0) {
        r->seed = random_seed();
    }
}

void request_resolve_lm_seed(MM3Request * r) {
    if (r->lm_seed < 0) {
        r->lm_seed = random_seed();
    }
}
