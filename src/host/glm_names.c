// src/host/glm_names.c - the tensor names, spelled once.  See the header for why.
#include "strata/host/glm_names.h"

#include <stddef.h>

/* A byte sink that refuses to run out of room silently: every emit fails, and the caller turns the failure
   into -1 rather than into a short name. */
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
    if (text == NULL) return 0;
    while (*text != '\0') {
        if (emit(s, *text++) != 0) return -1;
    }
    return 0;
}

/* Layers are 0..47 today and will not pass a 1e6 guard; a bigger number is a corrupt name, not a layer. */
static int emit_layer(sink *s, int value) {
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
    if (s->left == 0) return -1; /* no room for the terminator: the name would be truncated */
    *s->p = '\0';
    return (int)(cap - s->left);
}

int strata_blk_prefix(char *out, size_t cap, int layer) {
    sink s;
    if (out == NULL) return -1;
    s.p = out;
    s.left = cap;
    if (emit_text(&s, "blk.") != 0) return -1;
    if (emit_layer(&s, layer) != 0) return -1;
    if (emit(&s, '.') != 0) return -1;
    return finish(&s, cap);
}

int strata_blk_name(char *out, size_t cap, int layer, const char *suffix) {
    sink s;
    if (out == NULL) return -1;
    s.p = out;
    s.left = cap;
    if (emit_text(&s, "blk.") != 0) return -1;
    if (emit_layer(&s, layer) != 0) return -1;
    if (emit(&s, '.') != 0) return -1;
    if (emit_text(&s, suffix) != 0) return -1;
    return finish(&s, cap);
}

int strata_parse_blk(const char *name, int *layer, const char **suffix) {
    const char *p;
    int value = 0, digits = 0;
    if (name == NULL) return -1;
    if (name[0] != 'b' || name[1] != 'l' || name[2] != 'k' || name[3] != '.') return 0; /* not a block name */
    p = name + 4;
    while (*p >= '0' && *p <= '9') {
        value = value * 10 + (*p - '0');
        if (value > 1000000) return -1;
        ++p;
        ++digits;
    }
    if (digits == 0 || *p != '.') return -1;
    if (p[1] == '\0') return -1; /* "blk.3." names no tensor */
    if (layer != NULL) *layer = value;
    if (suffix != NULL) *suffix = p + 1;
    return 1;
}
