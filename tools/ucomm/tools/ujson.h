/* ujson.h - a small JSON reader and writer for the ucomm tools (C11, no dependencies).
 * Reader: objects, arrays, strings (escapes incl. \uXXXX as UTF-8), numbers, true/false/null. Values are a tree of
 * uj_node; free the root with uj_free. Writer: uj_buf with helpers that escape strings. */
#ifndef UJSON_H_
#define UJSON_H_

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { UJ_NULL, UJ_BOOL, UJ_NUM, UJ_STR, UJ_ARR, UJ_OBJ } uj_type;
typedef struct uj_node {
  uj_type type;
  double num;
  int boolean;
  char *str;             /* UJ_STR value */
  char *key;             /* set when the node is an object member */
  struct uj_node *child; /* first element / member */
  struct uj_node *next;
} uj_node;

static void uj_free(uj_node *n) {
  while (n) {
    uj_node *next = n->next;
    uj_free(n->child);
    free(n->str);
    free(n->key);
    free(n);
    n = next;
  }
}

typedef struct { const char *p, *end; const char *err; } uj_in;

static void uj_ws(uj_in *in) {
  while (in->p < in->end && (*in->p == ' ' || *in->p == '\t' || *in->p == '\n' || *in->p == '\r')) in->p++;
}
static char *uj_string(uj_in *in) {
  if (in->p >= in->end || *in->p != '"') { in->err = "expected string"; return NULL; }
  in->p++;
  size_t cap = 32, n = 0;
  char *s = malloc(cap);
  while (in->p < in->end && *in->p != '"') {
    unsigned c = (unsigned char)*in->p++;
    char tmp[4];
    size_t k = 1;
    if (c == '\\') {
      if (in->p >= in->end) break;
      char e = *in->p++;
      switch (e) {
        case 'n': tmp[0] = '\n'; break;
        case 't': tmp[0] = '\t'; break;
        case 'r': tmp[0] = '\r'; break;
        case 'b': tmp[0] = '\b'; break;
        case 'f': tmp[0] = '\f'; break;
        case 'u': {
          unsigned v = 0;
          for (int i = 0; i < 4 && in->p < in->end; i++) {
            char h = *in->p++;
            v = v * 16 + (unsigned)(h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h - '0');
          }
          if (v < 0x80) tmp[0] = (char)v;
          else if (v < 0x800) { tmp[0] = (char)(0xc0 | v >> 6); tmp[1] = (char)(0x80 | (v & 63)); k = 2; }
          else { tmp[0] = (char)(0xe0 | v >> 12); tmp[1] = (char)(0x80 | ((v >> 6) & 63)); tmp[2] = (char)(0x80 | (v & 63)); k = 3; }
          break;
        }
        default: tmp[0] = e;
      }
    } else tmp[0] = (char)c;
    if (n + k + 1 > cap) s = realloc(s, cap *= 2);
    memcpy(s + n, tmp, k);
    n += k;
  }
  if (in->p >= in->end) { free(s); in->err = "unterminated string"; return NULL; }
  in->p++;
  s[n] = 0;
  return s;
}
static uj_node *uj_value(uj_in *in, int depth) {
  uj_ws(in);
  if (in->p >= in->end || depth > 64) { in->err = "unexpected end"; return NULL; }
  uj_node *v = calloc(1, sizeof *v);
  char c = *in->p;
  if (c == '{' || c == '[') {
    v->type = c == '{' ? UJ_OBJ : UJ_ARR;
    in->p++;
    uj_node **tail = &v->child;
    uj_ws(in);
    if (in->p < in->end && *in->p == (c == '{' ? '}' : ']')) { in->p++; return v; }
    for (;;) {
      char *key = NULL;
      if (v->type == UJ_OBJ) {
        uj_ws(in);
        if (!(key = uj_string(in))) break;
        uj_ws(in);
        if (in->p >= in->end || *in->p++ != ':') { free(key); in->err = "expected ':'"; break; }
      }
      uj_node *m = uj_value(in, depth + 1);
      if (!m) { free(key); break; }
      m->key = key;
      *tail = m;
      tail = &m->next;
      uj_ws(in);
      if (in->p < in->end && *in->p == ',') { in->p++; continue; }
      if (in->p < in->end && *in->p == (v->type == UJ_OBJ ? '}' : ']')) { in->p++; return v; }
      in->err = "expected ',' or closing bracket";
      break;
    }
    uj_free(v);
    return NULL;
  }
  if (c == '"') {
    v->type = UJ_STR;
    if (!(v->str = uj_string(in))) { free(v); return NULL; }
    return v;
  }
  if (!strncmp(in->p, "true", 4)) { v->type = UJ_BOOL; v->boolean = 1; in->p += 4; return v; }
  if (!strncmp(in->p, "false", 5)) { v->type = UJ_BOOL; in->p += 5; return v; }
  if (!strncmp(in->p, "null", 4)) { v->type = UJ_NULL; in->p += 4; return v; }
  char *end = NULL;
  v->num = strtod(in->p, &end);
  if (end == in->p) { free(v); in->err = "invalid value"; return NULL; }
  v->type = UJ_NUM;
  in->p = end;
  return v;
}
/* Parses text[0..len); NULL on error with *err set. */
static uj_node *uj_parse(const char *text, size_t len, const char **err) {
  uj_in in = {text, text + len, NULL};
  uj_node *v = uj_value(&in, 0);
  uj_ws(&in);
  if (v && in.p != in.end) { uj_free(v); v = NULL; in.err = "trailing characters"; }
  if (err) *err = in.err;
  return v;
}
static uj_node *uj_get(const uj_node *o, const char *key) {
  if (!o || o->type != UJ_OBJ) return NULL;
  for (uj_node *m = o->child; m; m = m->next)
    if (m->key && !strcmp(m->key, key)) return m;
  return NULL;
}
static const char *uj_str(const uj_node *o, const char *key, const char *def) {
  const uj_node *v = uj_get(o, key);
  return v && v->type == UJ_STR ? v->str : def;
}
static double uj_num(const uj_node *o, const char *key, double def) {
  const uj_node *v = uj_get(o, key);
  return v && v->type == UJ_NUM ? v->num : def;
}
static int uj_bool(const uj_node *o, const char *key, int def) {
  const uj_node *v = uj_get(o, key);
  return v && v->type == UJ_BOOL ? v->boolean : def;
}
static int uj_len(const uj_node *a) {
  int n = 0;
  for (const uj_node *m = a ? a->child : NULL; m; m = m->next) n++;
  return n;
}

/* ---- writer */
typedef struct { char *s; size_t n, cap; } uj_buf;
static void uj_put(uj_buf *b, const char *fmt, ...) {
  va_list ap;
  for (;;) {
    va_start(ap, fmt);
    int k = vsnprintf(b->s ? b->s + b->n : NULL, b->s ? b->cap - b->n : 0, fmt, ap);
    va_end(ap);
    if (k < 0) return;
    if (b->s && b->n + (size_t)k < b->cap) { b->n += (size_t)k; return; }
    b->cap = (b->cap + (size_t)k + 1) * 2;
    b->s = realloc(b->s, b->cap);
    if (!b->n) b->s[0] = 0;
  }
}
static void uj_put_str(uj_buf *b, const char *s) {
  uj_put(b, "\"");
  for (; s && *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') uj_put(b, "\\%c", c);
    else if (c == '\n') uj_put(b, "\\n");
    else if (c < 0x20) uj_put(b, "\\u%04x", c);
    else uj_put(b, "%c", c);
  }
  uj_put(b, "\"");
}
/* Writes a node back as compact JSON. */
static void uj_dump(uj_buf *b, const uj_node *v) {
  if (!v) { uj_put(b, "null"); return; }
  switch (v->type) {
    case UJ_NULL: uj_put(b, "null"); break;
    case UJ_BOOL: uj_put(b, v->boolean ? "true" : "false"); break;
    case UJ_NUM:
      if (v->num == (double)(int64_t)v->num) uj_put(b, "%lld", (long long)v->num);
      else uj_put(b, "%.17g", v->num);
      break;
    case UJ_STR: uj_put_str(b, v->str); break;
    case UJ_ARR:
    case UJ_OBJ:
      uj_put(b, v->type == UJ_OBJ ? "{" : "[");
      for (const uj_node *m = v->child; m; m = m->next) {
        if (m != v->child) uj_put(b, ",");
        if (v->type == UJ_OBJ) { uj_put_str(b, m->key); uj_put(b, ":"); }
        uj_dump(b, m);
      }
      uj_put(b, v->type == UJ_OBJ ? "}" : "]");
  }
}
#endif
