// src/host/ds41f_names.c - DeepSeek V4.1 Flash tensor names, spelled once. See the header.
#include "strata/host/ds41f_names.h"

#include <stddef.h>
#include <string.h>

typedef struct {
    char *p;
    size_t left;
} sink;

static int emit(sink *s, char c) {
    if (s->left == 0) return -1;
    *s->p++ = c;
    --s->left;
    return 0;
}

static int emit_text(sink *s, const char *text) {
    while (*text != '\0') {
        if (emit(s, *text++) != 0) return -1;
    }
    return 0;
}

static int emit_int(sink *s, int value) {
    char digits[12];
    int n = 0, i;
    unsigned int rest;
    if (value < 0) return -1;
    rest = (unsigned int)value;
    do {
        digits[n++] = (char)('0' + (int)(rest % 10u));
        rest /= 10u;
    } while (rest != 0u);
    for (i = n - 1; i >= 0; --i) {
        if (emit(s, digits[i]) != 0) return -1;
    }
    return 0;
}

static int finish(sink *s, size_t cap) {
    if (s->left == 0) return -1;
    *s->p = '\0';
    return (int)(cap - s->left - 1u);
}

static const char *proj_word(enum strata_ds41f_proj proj) {
    switch (proj) {
    case STRATA_DS41F_GATE: return "gate";
    case STRATA_DS41F_UP: return "up";
    case STRATA_DS41F_DOWN: return "down";
    default: return NULL;
    }
}

int strata_ds41f_source_w(enum strata_ds41f_proj proj) {
    switch (proj) {
    case STRATA_DS41F_GATE: return 1;
    case STRATA_DS41F_UP: return 3;
    case STRATA_DS41F_DOWN: return 2;
    default: return -1;
    }
}

int strata_ds41f_expert_source_name(char *out, size_t cap, int layer, int expert, enum strata_ds41f_proj proj, const char *part) {
    sink s = {out, cap};
    int w = strata_ds41f_source_w(proj);
    if (out == NULL || cap == 0 || part == NULL || w < 0 || layer < 0 || layer >= STRATA_DS41F_LAYERS || expert < 0 ||
        expert >= STRATA_DS41F_EXPERTS)
        return -1;
    if (emit_text(&s, "layers.") != 0 || emit_int(&s, layer) != 0 || emit_text(&s, ".ffn.experts.") != 0 || emit_int(&s, expert) != 0 ||
        emit_text(&s, ".w") != 0 || emit_int(&s, w) != 0 || emit(&s, '.') != 0 || emit_text(&s, part) != 0)
        return -1;
    return finish(&s, cap);
}

int strata_ds41f_expert_pack_name(char *out, size_t cap, int layer, enum strata_ds41f_proj proj) {
    sink s = {out, cap};
    const char *word = proj_word(proj);
    if (out == NULL || cap == 0 || word == NULL || layer < 0 || layer >= STRATA_DS41F_LAYERS) return -1;
    if (emit_text(&s, "blk.") != 0 || emit_int(&s, layer) != 0 || emit_text(&s, ".ffn_") != 0 || emit_text(&s, word) != 0 ||
        emit_text(&s, "_exps.weight") != 0)
        return -1;
    return finish(&s, cap);
}

int strata_ds41f_keep_name(char *out, size_t cap, int layer) {
    sink s = {out, cap};
    if (out == NULL || cap == 0 || layer < 0 || layer >= STRATA_DS41F_LAYERS) return -1;
    if (emit_text(&s, "blk.") != 0 || emit_int(&s, layer) != 0 || emit_text(&s, ".ffn_exps_keep") != 0) return -1;
    return finish(&s, cap);
}

int strata_ds41f_parse_pack_name(const char *name, int *layer, enum strata_ds41f_proj *proj) {
    const char *p;
    int value = 0, digits = 0;
    if (name == NULL || strncmp(name, "blk.", 4) != 0) return 0;
    p = name + 4;
    while (*p >= '0' && *p <= '9') {
        if (digits >= 6) return -1;
        value = value * 10 + (*p - '0');
        ++p;
        ++digits;
    }
    if (digits == 0 || strncmp(p, ".ffn_", 5) != 0) return strncmp(p, ".", 1) == 0 ? 0 : -1;
    p += 5;
    if (strcmp(p, "gate_exps.weight") == 0) *proj = STRATA_DS41F_GATE;
    else if (strcmp(p, "up_exps.weight") == 0) *proj = STRATA_DS41F_UP;
    else if (strcmp(p, "down_exps.weight") == 0) *proj = STRATA_DS41F_DOWN;
    else return 0;
    if (value >= STRATA_DS41F_LAYERS) return -1;
    *layer = value;
    return 1;
}

int strata_ds41f_engram_layer(int index) {
    static const int layers[STRATA_DS41F_ENGRAM_LAYER_COUNT] = {1, 14};
    if (index < 0 || index >= STRATA_DS41F_ENGRAM_LAYER_COUNT) return -1;
    return layers[index];
}

int strata_ds41f_is_engram_table(const char *name) {
    const char *p;
    if (name == NULL || strncmp(name, "layers.", 7) != 0) return 0;
    p = strstr(name, ".engram.embed.");
    return p != NULL && (strcmp(p, ".engram.embed.weight") == 0 || strcmp(p, ".engram.embed.scale") == 0);
}

int strata_ds41f_is_routed_expert(const char *name) {
    if (name == NULL || strncmp(name, "layers.", 7) != 0) return 0;
    return strstr(name, ".ffn.experts.") != NULL;
}
