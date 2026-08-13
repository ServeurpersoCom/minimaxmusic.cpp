// prompt.h: special token prompt assembly for the global LM
//
// The prompt is one token stream:
//   <|im_start|><|caption_start|>caption<|caption_end|>
//   <|lyrics_start|>lyrics<|lyrics_end|><|im_end|><|audio_start|>
// with the caption cleaned (special tags rewritten as "X is Y", markdown
// stripped) and the lyrics normalized (leading structure tags kept and
// lowercased, "[start]\n" prepended), mirroring the reference
// _clean_caption / _normalize_lyrics. The unconditional CFG stream keeps
// the first token and the two trailing ones and replaces everything else
// by the audio CFG token.
#pragma once

#include <cctype>
#include <string>
#include <vector>

// Special token ids (tokenizer/tokenizer.json added_tokens)
static const int MM3_IM_START      = 151644;
static const int MM3_IM_END        = 151645;
static const int MM3_AUDIO_CFG     = 151654;
static const int MM3_AUDIO_START   = 151669;
static const int MM3_AUDIO_END     = 151670;
static const int MM3_CAPTION_START = 151671;
static const int MM3_CAPTION_END   = 151672;
static const int MM3_LYRICS_START  = 151673;
static const int MM3_LYRICS_END    = 151674;

// Audio code space inside the LM vocab
static const int MM3_AUDIO_CODE_OFFSET = 151675;
static const int MM3_SEMANTIC_VOCAB    = 16384;

static std::string mm3_strip(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> mm3_split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   pos = 0;
    while (pos <= s.size()) {
        size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, nl - pos));
        pos = nl + 1;
    }
    return out;
}

static void mm3_replace_all(std::string & s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Rewrites every <|inner|> tag: "<|A B|>" -> "A is B", "<|A|>" -> "A"
static std::string mm3_rewrite_special_tags(const std::string & text) {
    std::string out;
    size_t      pos = 0;
    while (pos < text.size()) {
        size_t open = text.find("<|", pos);
        if (open == std::string::npos) {
            out += text.substr(pos);
            break;
        }
        size_t close = text.find("|>", open + 2);
        if (close == std::string::npos) {
            out += text.substr(pos);
            break;
        }
        out += text.substr(pos, open - pos);
        std::string inner = mm3_strip(text.substr(open + 2, close - open - 2));
        size_t      ws    = inner.find_first_of(" \t");
        if (ws != std::string::npos) {
            out += mm3_strip(inner.substr(0, ws)) + " is " + mm3_strip(inner.substr(ws + 1));
        } else {
            out += inner;
        }
        pos = close + 2;
    }
    return out;
}

// Strips markdown emphasis: "**x**" pairs first, then single "*x*" spans
static std::string mm3_strip_emphasis(std::string line) {
    while (true) {
        size_t a = line.find("**");
        if (a == std::string::npos) {
            break;
        }
        size_t b = line.find("**", a + 2);
        if (b == std::string::npos || b == a + 2) {
            break;
        }
        line = line.substr(0, a) + line.substr(a + 2, b - a - 2) + line.substr(b + 2);
    }
    std::string out;
    size_t      pos = 0;
    while (pos < line.size()) {
        if (line[pos] == '*') {
            size_t b = line.find_first_of("*\n", pos + 1);
            if (b != std::string::npos && line[b] == '*' && b > pos + 1) {
                out += line.substr(pos + 1, b - pos - 1);
                pos = b + 1;
                continue;
            }
        }
        out += line[pos++];
    }
    return out;
}

// Reference _clean_caption
static std::string mm3_clean_caption(const std::string & caption) {
    std::string text = mm3_rewrite_special_tags(caption);

    std::vector<std::string> lines = mm3_split_lines(text);
    std::string              joined;
    for (size_t li = 0; li < lines.size(); li++) {
        std::string line = lines[li];

        // Leading markdown header: up to 3 spaces, 1..6 '#', whitespace
        size_t p = 0;
        while (p < line.size() && p < 3 && line[p] == ' ') {
            p++;
        }
        size_t h = p;
        while (h < line.size() && h < p + 6 && line[h] == '#') {
            h++;
        }
        if (h > p && h < line.size() && (line[h] == ' ' || line[h] == '\t')) {
            size_t c = h;
            while (c < line.size() && (line[c] == ' ' || line[c] == '\t')) {
                c++;
            }
            line = line.substr(c);
        }

        // Leading list marker: optional spaces, one of * + -, whitespace
        p = line.find_first_not_of(" \t");
        if (p != std::string::npos && (line[p] == '*' || line[p] == '+' || line[p] == '-') && p + 1 < line.size() &&
            (line[p + 1] == ' ' || line[p + 1] == '\t')) {
            size_t c = p + 1;
            while (c < line.size() && (line[c] == ' ' || line[c] == '\t')) {
                c++;
            }
            line = line.substr(c);
        }

        line = mm3_strip_emphasis(line);

        // Trim trailing whitespace
        size_t e = line.find_last_not_of(" \t\r");
        line     = (e == std::string::npos) ? "" : line.substr(0, e + 1);

        // Horizontal rule line: only - * _ (3+) and spaces
        std::string bare = mm3_strip(line);
        if (bare.size() >= 3 && bare.find_first_not_of("-*_") == std::string::npos) {
            line = "";
        }

        joined += line;
        if (li + 1 < lines.size()) {
            joined += "\n";
        }
    }

    mm3_replace_all(joined, "\xE2\x80\xA2 ", "");  // bullet dot
    mm3_replace_all(joined, "    ", "");
    while (joined.find("\n\n") != std::string::npos) {
        mm3_replace_all(joined, "\n\n", "\n");
    }
    return joined;
}

// Reference _normalize_lyrics
static std::string mm3_normalize_lyrics(const std::string & lyrics) {
    std::vector<std::string> lines = mm3_split_lines(lyrics);
    std::string              text;
    for (size_t li = 0; li < lines.size(); li++) {
        const std::string & line = lines[li];

        // Leading run of [tag] groups separated by spaces or tabs
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) {
            p++;
        }
        size_t tag_end = std::string::npos;
        size_t q       = p;
        while (q < line.size() && line[q] == '[') {
            size_t close = line.find(']', q + 1);
            if (close == std::string::npos || close == q + 1) {
                break;
            }
            tag_end = close + 1;
            q       = close + 1;
            while (q < line.size() && (line[q] == ' ' || line[q] == '\t')) {
                q++;
            }
        }
        if (tag_end != std::string::npos) {
            text += mm3_strip(line.substr(p, tag_end - p));
        } else {
            text += line;
        }
        if (li + 1 < lines.size()) {
            text += "\n";
        }
    }

    mm3_replace_all(text, "] ", "]\n");
    mm3_replace_all(text, " [", "\n[");
    mm3_replace_all(text, " ^ ", "\n");

    // Lowercase inside brackets
    bool in_tag = false;
    for (char & c : text) {
        if (c == '[') {
            in_tag = true;
        } else if (c == ']') {
            in_tag = false;
        } else if (in_tag) {
            c = (char) tolower((unsigned char) c);
        }
    }
    return "[start]\n" + text;
}

// Assembles the conditional token stream. The unconditional stream is the
// same with indices [1, n-3] replaced by MM3_AUDIO_CFG.
template <typename Tokenize>
static std::vector<int> mm3_build_prompt_ids(Tokenize            bpe_encode,
                                             const std::string & caption,
                                             const std::string & lyrics) {
    std::vector<int> ids;
    ids.push_back(MM3_IM_START);
    ids.push_back(MM3_CAPTION_START);
    for (int id : bpe_encode(mm3_clean_caption(caption))) {
        ids.push_back(id);
    }
    ids.push_back(MM3_CAPTION_END);
    ids.push_back(MM3_LYRICS_START);
    for (int id : bpe_encode(mm3_normalize_lyrics(lyrics))) {
        ids.push_back(id);
    }
    ids.push_back(MM3_LYRICS_END);
    ids.push_back(MM3_IM_END);
    ids.push_back(MM3_AUDIO_START);
    return ids;
}
