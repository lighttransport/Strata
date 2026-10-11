// src/host/glm_env.c - every tuning knob in one table.  See the header for why.
#include "strata/host/glm_env.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The decoder's table is 133 knobs; 512 leaves room for the other binaries' knobs to be merged later
   without a second capacity decision. */
#define STRATA_ENV_KNOB_MAX 512

typedef struct {
    strata_env_knob knob;
    double value;                       /* INT and FLAG keep their number here too */
    char text[STRATA_ENV_TEXT_MAX];
} entry;

static entry g_table[STRATA_ENV_KNOB_MAX];
static size_t g_count;
static int g_last_error = STRATA_ENV_ERR_UNINIT;

/* "<name>=<value> is set but <what>" / "<name>: <what>" when there is no value to show.  What the caller is
   told is the variable and the value, because the value is what the operator typed. */
static int report(char *err, size_t cap, int code, const char *name, const char *value, const char *what) {
    if (err != NULL && cap != 0) {
        if (value != NULL) (void)snprintf(err, cap, "%s=%s is set but %s", name, value, what);
        else (void)snprintf(err, cap, "%s: %s", name, what);
    }
    return code;
}

/* An INT is parsed with strtol in base 10, not strtod: strtod accepts a hexadecimal 0x8 as 8, while the
   engine's `std::stoi` read it as 0 with junk after it, which was an error. */
static int parse_int(const char *text, double *out) {
    char *end = NULL;
    long value;
    if (text == NULL || *text == '\0') return 0;
    errno = 0;
    value = strtol(text, &end, 10);
    if (end == text || end == NULL || *end != '\0') return 0; /* trailing junk is not a number */
    if (errno == ERANGE) return 0;
    *out = (double)value;
    return 1;
}

static int parse_float(const char *text, double *out) {
    char *end = NULL;
    double value;
    if (text == NULL || *text == '\0') return 0;
    value = strtod(text, &end);
    if (end == text || end == NULL || *end != '\0') return 0; /* trailing junk is not a number */
    if (isnan(value) || value > 1e18 || value < -1e18) return 0;
    *out = value;
    return 1;
}

static const char *kind_name(strata_env_kind kind) {
    switch (kind) {
    case STRATA_ENV_INT: return "it is not an integer";
    case STRATA_ENV_FLOAT: return "it is not a number";
    case STRATA_ENV_FLAG:
    case STRATA_ENV_FLAG_LOOSE: return "it is neither 0 nor 1";
    case STRATA_ENV_TEXT: return "it is not text";
    }
    return "it is not a value";
}

static char *copy_text(entry *e, const char *value) {
    size_t n = strlen(value);
    if (n >= sizeof e->text) return NULL;
    memcpy(e->text, value, n + 1);
    return e->text;
}

static int load_one(entry *e, char *err, size_t cap) {
    const char *value = getenv(e->knob.name);
    char what[192];
    if (value == NULL) {
        e->value = e->knob.def;
        e->text[0] = '\0';
        return STRATA_ENV_OK;
    }
    switch (e->knob.kind) {
    case STRATA_ENV_INT:
    case STRATA_ENV_FLOAT: {
        double parsed = 0.0;
        int ok = e->knob.kind == STRATA_ENV_INT ? parse_int(value, &parsed) : parse_float(value, &parsed);
        if (!ok) {
            (void)snprintf(what, sizeof what, "%s", kind_name(e->knob.kind));
            return report(err, cap, STRATA_ENV_ERR_FORMAT, e->knob.name, value, what);
        }
        if (parsed < e->knob.lo || parsed > e->knob.hi) {
            if (e->knob.lo == e->knob.hi) (void)snprintf(what, sizeof what, "it must be %.17g", e->knob.lo);
            else (void)snprintf(what, sizeof what, "it must be %.17g..%.17g", e->knob.lo, e->knob.hi);
            return report(err, cap, STRATA_ENV_ERR_RANGE, e->knob.name, value, what);
        }
        e->value = parsed;
        e->text[0] = '\0';
        return STRATA_ENV_OK;
    }
    case STRATA_ENV_FLAG:
        if (strcmp(value, "0") == 0 || strcmp(value, "1") == 0) {
            e->value = value[0] == '1' ? 1.0 : 0.0;
            return STRATA_ENV_OK;
        }
        return report(err, cap, STRATA_ENV_ERR_FORMAT, e->knob.name, value, "it is neither 0 nor 1");
    case STRATA_ENV_FLAG_LOOSE:
        e->value = strcmp(value, "1") == 0 ? 1.0 : 0.0;
        return STRATA_ENV_OK;
    case STRATA_ENV_TEXT:
        if (copy_text(e, value) == NULL) {
            if (err != NULL && cap != 0)
                (void)snprintf(err, cap, "%s is longer than %d bytes", e->knob.name, STRATA_ENV_TEXT_MAX);
            return STRATA_ENV_ERR_TEXT;
        }
        return STRATA_ENV_OK;
    }
    return STRATA_ENV_ERR_FORMAT;
}

int strata_env_init(const strata_env_knob *knobs, size_t count, char *err, size_t cap) {
    size_t i;
    if (err != NULL && cap != 0) err[0] = '\0';
    if (count > STRATA_ENV_KNOB_MAX) {
        (void)report(err, cap, STRATA_ENV_ERR_ARGS, "strata_env_init", NULL, "the knob table is too large");
        return STRATA_ENV_ERR_ARGS;
    }
    if (knobs == NULL && count != 0) {
        (void)report(err, cap, STRATA_ENV_ERR_ARGS, "strata_env_init", NULL, "the knob table is null");
        return STRATA_ENV_ERR_ARGS;
    }
    /* Against the NEW table, not the one being replaced: a previous table must not make a valid name look
       like a duplicate. */
    for (i = 0; i < count; ++i) {
        size_t j;
        for (j = 0; j < i; ++j) {
            if (knobs[i].name != NULL && knobs[j].name != NULL && strcmp(knobs[i].name, knobs[j].name) == 0) {
                (void)report(err, cap, STRATA_ENV_ERR_DUP, knobs[i].name, NULL, "two knobs are named this");
                return STRATA_ENV_ERR_DUP;
            }
        }
    }
    g_count = count;
    g_last_error = STRATA_ENV_OK;
    for (i = 0; i < count; ++i) {
        int status;
        g_table[i].knob = knobs[i];
        g_table[i].value = knobs[i].def;
        g_table[i].text[0] = '\0';
        status = load_one(&g_table[i], err, cap);
        if (status != STRATA_ENV_OK) {
            g_last_error = status;
            return status;
        }
    }
    return STRATA_ENV_OK;
}

size_t strata_env_count(void) { return g_count; }

int strata_env_id(const char *name) {
    size_t i;
    if (name == NULL) return -1;
    for (i = 0; i < g_count; ++i) {
        if (g_table[i].knob.name != NULL && strcmp(g_table[i].knob.name, name) == 0) return (int)i;
    }
    return -1;
}

int strata_env_last_error(void) { return g_last_error; }

int strata_env_get_int(int id) {
    if (g_count == 0) {
        g_last_error = STRATA_ENV_ERR_UNINIT;
        return 0;
    }
    if (id < 0 || (size_t)id >= g_count) {
        g_last_error = STRATA_ENV_ERR_ID;
        return 0;
    }
    g_last_error = STRATA_ENV_OK;
    if (g_table[id].knob.kind == STRATA_ENV_TEXT) return g_table[id].text[0] == '\0' ? 0 : 1;
    return (int)g_table[id].value;
}

double strata_env_get(int id) {
    if (g_count == 0 || id < 0 || (size_t)id >= g_count) {
        g_last_error = g_count == 0 ? STRATA_ENV_ERR_UNINIT : STRATA_ENV_ERR_ID;
        return 0.0;
    }
    g_last_error = STRATA_ENV_OK;
    return g_table[id].value;
}

const char *strata_env_text(int id) {
    if (g_count == 0 || id < 0 || (size_t)id >= g_count) {
        g_last_error = g_count == 0 ? STRATA_ENV_ERR_UNINIT : STRATA_ENV_ERR_ID;
        return NULL;
    }
    g_last_error = STRATA_ENV_OK;
    if (g_table[id].knob.kind != STRATA_ENV_TEXT) return NULL;
    return g_table[id].text;
}

/* ---- The historical switch shapes, kept exactly as the call sites had them ---------------------------- */

int strata_env_truthy_number(const char *name) {
    const char *v = getenv(name);
    return v != NULL && atoi(v) != 0;
}

int strata_env_truthy_one(const char *name) {
    const char *v = getenv(name);
    return v != NULL && v[0] == '1';
}

int strata_env_truthy_default(const char *name, int when_unset) {
    const char *v = getenv(name);
    return v == NULL ? when_unset : v[0] != '0';
}

int strata_env_truthy_nonempty(const char *name) {
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0' && v[0] != '0';
}

int strata_env_is_one(const char *name) {
    const char *v = getenv(name);
    return v != NULL && strcmp(v, "1") == 0;
}

double strata_env_value_double(const char *name, double when_unset) {
    const char *v = getenv(name);
    return v == NULL ? when_unset : atof(v);
}

long long strata_env_value_ll(const char *name, long long when_unset) {
    const char *v = getenv(name);
    return v == NULL ? when_unset : atoll(v);
}

int strata_env_is_set(const char *name) { return getenv(name) != NULL; }

int strata_env_truthy_number_default(const char *name, int when_unset) {
    const char *v = getenv(name);
    return v == NULL ? when_unset : atoi(v) != 0;
}

int strata_env_equals(const char *name, const char *text) {
    const char *v = getenv(name);
    return v != NULL && text != NULL && strcmp(v, text) == 0;
}

int strata_env_text_on(const char *name) {
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0;
}

int strata_env_int_or(const char *name, int fallback) {
    const char *v = getenv(name);
    return (v != NULL && v[0] != '\0') ? atoi(v) : fallback;
}
