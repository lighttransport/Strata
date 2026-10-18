/* pf_plan.c - manifest parsing and the cooperative read plan for strata-pfetch (see pf_plan.h). */
#define _GNU_SOURCE
#include "pf_plan.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GAP_MERGE (1u << 20) /* read across holes smaller than this inside one block */

static int fail(char *err, size_t errn, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, errn, fmt, ap);
  va_end(ap);
  return -1;
}

static void *grow(void *p, size_t *cap, size_t need, size_t elem) {
  if (need <= *cap) return p;
  size_t c = *cap ? *cap : 64;
  while (c < need) c *= 2;
  void *q = realloc(p, c * elem);
  if (!q) {
    fprintf(stderr, "pfetch: out of memory\n");
    exit(3);
  }
  *cap = c;
  return q;
}

/* Expand "@L/" and "{rank}" / "{rank2}". */
char *pf_expand(const char *s, const char *local_root, int rank, char *err, size_t errn) {
  char buf[4096];
  size_t o = 0;
  if (strncmp(s, "@L/", 3) == 0) {
    if (!local_root) {
      fail(err, errn, "path %s uses @L/ but no --local-root was given", s);
      return NULL;
    }
    o = (size_t)snprintf(buf, sizeof buf, "%s/", local_root);
    s += 3;
  }
  while (*s && o + 16 < sizeof buf) {
    if (strncmp(s, "{rank2}", 7) == 0) {
      if (rank < 0) goto norank;
      o += (size_t)snprintf(buf + o, sizeof buf - o, "%02d", rank);
      s += 7;
    } else if (strncmp(s, "{rank}", 6) == 0) {
      if (rank < 0) goto norank;
      o += (size_t)snprintf(buf + o, sizeof buf - o, "%d", rank);
      s += 6;
    } else
      buf[o++] = *s++;
  }
  if (*s) {
    fail(err, errn, "path too long");
    return NULL;
  }
  buf[o] = 0;
  return strdup(buf);
norank:
  fail(err, errn, "{rank} in a path that is not per rank: %s", s);
  return NULL;
}

static uint32_t str_hash(const char *s) {
  uint32_t h = 2166136261u;
  while (*s) h = (h ^ (uint8_t)*s++) * 16777619u;
  return h;
}

static uint32_t intern_src(pf_plan *p, const char *path) {
  if (p->nsrc * 2 + 2 > p->hash_cap) {
    uint32_t cap = p->hash_cap ? p->hash_cap * 2 : 256;
    uint32_t *h = malloc(cap * sizeof *h);
    for (uint32_t i = 0; i < cap; i++) h[i] = UINT32_MAX;
    for (uint32_t i = 0; i < p->nsrc; i++) {
      uint32_t k = str_hash(p->srcs[i]) & (cap - 1);
      while (h[k] != UINT32_MAX) k = (k + 1) & (cap - 1);
      h[k] = i;
    }
    free(p->src_hash);
    p->src_hash = h;
    p->hash_cap = cap;
  }
  uint32_t k = str_hash(path) & (p->hash_cap - 1);
  while (p->src_hash[k] != UINT32_MAX) {
    if (strcmp(p->srcs[p->src_hash[k]], path) == 0) return p->src_hash[k];
    k = (k + 1) & (p->hash_cap - 1);
  }
  size_t cap = p->cap_src;
  p->srcs = grow(p->srcs, &cap, p->nsrc + 1, sizeof *p->srcs);
  p->cap_src = (uint32_t)cap;
  p->srcs[p->nsrc] = strdup(path);
  p->src_hash[k] = p->nsrc;
  return p->nsrc++;
}

static int parse_u64(const char *s, uint64_t *v) {
  char *e;
  if (!s || !*s || *s == '-') return -1;
  *v = strtoull(s, &e, 10);
  return *e ? -1 : 0;
}

int pf_parse(pf_plan *p, const char *text, size_t n, int ranks, const char *local_root, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  p->ranks = ranks;
  p->block = 16u << 20;
  size_t cap_group = 0, cap_dest = 0, cap_tmpl = 0, cap_link = 0, cap_copy = 0;
  size_t lineno = 0, pos = 0;
  int seen_header = 0;
  char line[16384];
  while (pos < n) {
    size_t e = pos;
    while (e < n && text[e] != '\n') e++;
    size_t len = e - pos;
    lineno++;
    if (len >= sizeof line) return fail(err, errn, "manifest line %zu too long", lineno);
    memcpy(line, text + pos, len);
    line[len] = 0;
    pos = e + 1;
    char *tok[10];
    int nt = 0;
    for (char *s = strtok(line, " \t\r"); s && nt < 10; s = strtok(NULL, " \t\r")) tok[nt++] = s;
    if (nt == 0 || tok[0][0] == '#') continue;
    if (!seen_header) {
      if (strcmp(tok[0], "PFETCH") != 0 || nt < 2 || strcmp(tok[1], "1") != 0)
        return fail(err, errn, "manifest must start with 'PFETCH 1'");
      for (int i = 2; i < nt; i++)
        if (strncmp(tok[i], "block=", 6) == 0 && (parse_u64(tok[i] + 6, &p->block) || p->block < 4096))
          return fail(err, errn, "bad block size");
      seen_header = 1;
      continue;
    }
    char kind = tok[0][0];
    if (kind == 'G' && nt == 4) {
      uint64_t gid;
      if (parse_u64(tok[1], &gid) || gid != p->ngroup) return fail(err, errn, "line %zu: group ids must be 0,1,2,...", lineno);
      p->groups = grow(p->groups, &cap_group, p->ngroup + 1, sizeof *p->groups);
      p->groups[p->ngroup].name = strdup(tok[2]);
      p->groups[p->ngroup].prio = atoi(tok[3]);
      p->ngroup++;
    } else if (kind == 'D' && (nt == 6 || nt == 7)) {
      uint64_t did, size;
      if (parse_u64(tok[1], &did) || did != p->ntmpl) return fail(err, errn, "line %zu: destination ids must be 0,1,2,...", lineno);
      if (parse_u64(tok[4], &size)) return fail(err, errn, "line %zu: bad size", lineno);
      int mode = strcmp(tok[5], "atomic") == 0 ? PF_ATOMIC : strcmp(tok[5], "sparse") == 0 ? PF_SPARSE : -1;
      if (mode < 0) return fail(err, errn, "line %zu: mode must be sparse or atomic", lineno);
      const char *sidecar = NULL;
      if (nt == 7) {
        if (strncmp(tok[6], "sidecar=", 8) != 0) return fail(err, errn, "line %zu: unknown field %s", lineno, tok[6]);
        sidecar = tok[6] + 8;
      }
      int all = strcmp(tok[2], "*") == 0, r0 = 0;
      if (!all) {
        uint64_t r;
        if (parse_u64(tok[2], &r) || r >= (uint64_t)ranks) return fail(err, errn, "line %zu: rank out of range", lineno);
        r0 = (int)r;
      }
      p->tmpl_first = grow(p->tmpl_first, &cap_tmpl, p->ntmpl + 1, sizeof *p->tmpl_first);
      p->tmpl_count = realloc(p->tmpl_count, cap_tmpl * sizeof *p->tmpl_count);
      p->tmpl_first[p->ntmpl] = p->ndest;
      p->tmpl_count[p->ntmpl] = all ? (uint32_t)ranks : 1;
      for (int k = 0; k < (all ? ranks : 1); k++) {
        int r = all ? k : r0;
        p->dests = grow(p->dests, &cap_dest, p->ndest + 1, sizeof *p->dests);
        pf_dest *d = &p->dests[p->ndest];
        memset(d, 0, sizeof *d);
        d->rank = r;
        d->size = size;
        d->mode = mode;
        if (!(d->path = pf_expand(tok[3], local_root, r, err, errn))) return -1;
        if (sidecar && !(d->sidecar = pf_expand(sidecar, local_root, r, err, errn))) return -1;
        p->ndest++;
      }
      p->ntmpl++;
    } else if (kind == 'S' && nt == 4) {
      int all = strcmp(tok[1], "*") == 0, r0 = 0;
      if (!all) {
        uint64_t r;
        if (parse_u64(tok[1], &r) || r >= (uint64_t)ranks) return fail(err, errn, "line %zu: rank out of range", lineno);
        r0 = (int)r;
      }
      for (int k = 0; k < (all ? ranks : 1); k++) {
        int r = all ? k : r0;
        p->links = grow(p->links, &cap_link, p->nlink + 1, sizeof *p->links);
        pf_link *l = &p->links[p->nlink++];
        l->rank = r;
        if (!(l->link = pf_expand(tok[2], local_root, r, err, errn)) || !(l->target = pf_expand(tok[3], local_root, r, err, errn)))
          return -1;
      }
    } else if (kind == 'C' && (nt == 6 || nt == 7)) {
      uint64_t did, doff, soff, len, gid = 0;
      if (parse_u64(tok[1], &did) || did >= p->ntmpl) return fail(err, errn, "line %zu: unknown destination", lineno);
      if (parse_u64(tok[2], &doff) || parse_u64(tok[4], &soff) || parse_u64(tok[5], &len) || (nt == 7 && parse_u64(tok[6], &gid)))
        return fail(err, errn, "line %zu: bad number", lineno);
      if (len == 0) continue;
      if (strstr(tok[3], "{rank") || strncmp(tok[3], "@L/", 3) == 0)
        return fail(err, errn, "line %zu: sources must be shared paths", lineno);
      p->copies = grow(p->copies, &cap_copy, p->ncopy + 1, sizeof *p->copies);
      pf_copy *c = &p->copies[p->ncopy++];
      c->tmpl = (uint32_t)did;
      c->dst_off = doff;
      c->src = intern_src(p, tok[3]);
      c->src_off = soff;
      c->len = len;
      c->group = (uint32_t)gid;
    } else
      return fail(err, errn, "line %zu: malformed record '%s'", lineno, tok[0]);
  }
  if (!seen_header) return fail(err, errn, "empty manifest");
  if (p->ngroup == 0) {
    p->groups = grow(p->groups, &cap_group, 1, sizeof *p->groups);
    p->groups[0].name = strdup("all");
    p->groups[0].prio = 0;
    p->ngroup = 1;
  }
  for (size_t i = 0; i < p->ncopy; i++)
    if (p->copies[i].group >= p->ngroup) return fail(err, errn, "copy %zu: unknown group", i);
  return 0;
}

static int cmp_piece(const void *a, const void *b) {
  const pf_piece *x = a, *y = b;
#define C3(f) \
  if (x->f != y->f) return x->f < y->f ? -1 : 1;
  C3(src) C3(blkno) C3(rank) C3(src_off) C3(dest) C3(dst_off)
#undef C3
  return 0;
}

static int cmp_dst(const void *a, const void *b) {
  const pf_piece *x = *(const pf_piece *const *)a, *y = *(const pf_piece *const *)b;
  if (x->dest != y->dest) return x->dest < y->dest ? -1 : 1;
  return x->dst_off < y->dst_off ? -1 : x->dst_off > y->dst_off;
}

static int cmp_span(const void *a, const void *b) {
  const pf_span *x = a, *y = b;
  return x->off < y->off ? -1 : x->off > y->off;
}

static int least_loaded(const uint64_t *load, int ranks, int rot) {
  int best = rot % ranks;
  for (int k = 1; k < ranks; k++) {
    int r = (rot + k) % ranks;
    if (load[r] < load[best]) best = r;
  }
  return best;
}

typedef struct { uint64_t a, b; uint32_t i; } key3;
static int cmp_key3(const void *x, const void *y) {
  const key3 *p = x, *q = y;
  if (p->a != q->a) return p->a < q->a ? -1 : 1;
  if (p->b != q->b) return p->b < q->b ? -1 : 1;
  return p->i < q->i ? -1 : p->i > q->i;
}

int pf_build(pf_plan *p, int assign, char *err, size_t errn) {
  const uint64_t B = p->block;
  size_t cap = 0, n = 0;
  pf_piece *pc = NULL;
  for (size_t i = 0; i < p->ncopy; i++) {
    const pf_copy *c = &p->copies[i];
    for (uint32_t k = 0; k < p->tmpl_count[c->tmpl]; k++) {
      uint32_t d = p->tmpl_first[c->tmpl] + k;
      if (c->dst_off + c->len > p->dests[d].size || c->dst_off + c->len < c->dst_off)
        return fail(err, errn, "copy into %s ends past its size", p->dests[d].path);
      if (p->dests[d].skip) continue;
      for (uint64_t o = 0; o < c->len;) {
        uint64_t s = c->src_off + o, take = B - s % B;
        if (take > c->len - o) take = c->len - o;
        pc = grow(pc, &cap, n + 1, sizeof *pc);
        pf_piece *q = &pc[n++];
        memset(q, 0, sizeof *q);
        q->src = c->src;
        q->blkno = s / B;
        q->src_off = s;
        q->dst_off = c->dst_off + o;
        q->len = take;
        q->dest = d;
        q->group = c->group;
        q->rank = p->dests[d].rank;
        o += take;
      }
    }
  }
  qsort(pc, n, sizeof *pc, cmp_piece);
  p->pieces = pc;
  p->npiece = n;

  /* destination checks: no overlapping writes; atomic files tiled exactly */
  pf_piece **byd = malloc((n ? n : 1) * sizeof *byd);
  for (size_t i = 0; i < n; i++) byd[i] = &pc[i];
  qsort(byd, n, sizeof *byd, cmp_dst);
  for (uint32_t d = 0; d < p->ndest; d++) p->dests[d].covered = 0;
  for (size_t i = 0; i < n; i++) {
    pf_dest *d = &p->dests[byd[i]->dest];
    if (i + 1 < n && byd[i + 1]->dest == byd[i]->dest && byd[i]->dst_off + byd[i]->len > byd[i + 1]->dst_off) {
      uint64_t at = byd[i + 1]->dst_off;
      free(byd);
      return fail(err, errn, "overlapping copies into %s at %" PRIu64, d->path, at);
    }
    d->covered += byd[i]->len;
  }
  free(byd);
  for (uint32_t d = 0; d < p->ndest; d++)
    if (p->dests[d].mode == PF_ATOMIC && !p->dests[d].skip && p->dests[d].covered != p->dests[d].size)
      return fail(err, errn, "atomic destination %s: copies cover %" PRIu64 " of %" PRIu64 " bytes", p->dests[d].path,
                  p->dests[d].covered, p->dests[d].size);

  /* blocks, read spans (union with small gaps merged), messages (exact union per destination rank) */
  size_t cap_b = 0, cap_r = 0, cap_m = 0, cap_ms = 0;
  pf_span *tmp = NULL;
  size_t cap_t = 0;
  p->total_read = p->total_needed = 0;
  for (size_t i = 0; i < n;) {
    size_t j = i;
    while (j < n && pc[j].src == pc[i].src && pc[j].blkno == pc[i].blkno) j++;
    p->blocks = grow(p->blocks, &cap_b, p->nblock + 1, sizeof *p->blocks);
    pf_block *b = &p->blocks[p->nblock];
    memset(b, 0, sizeof *b);
    b->src = pc[i].src;
    b->index = pc[i].blkno;
    b->piece0 = (uint32_t)i;
    b->npiece = (uint32_t)(j - i);
    b->prio = INT32_MAX;
    tmp = grow(tmp, &cap_t, j - i, sizeof *tmp);
    for (size_t k = i; k < j; k++) {
      tmp[k - i].off = pc[k].src_off;
      tmp[k - i].len = pc[k].len;
      pc[k].block = (uint32_t)p->nblock;
      if (p->groups[pc[k].group].prio < b->prio) b->prio = p->groups[pc[k].group].prio;
    }
    qsort(tmp, j - i, sizeof *tmp, cmp_span);
    b->rspan0 = (uint32_t)p->nrspan;
    for (size_t k = 0; k < j - i; k++) {
      pf_span *last = b->nrspan ? &p->rspans[p->nrspan - 1] : NULL;
      if (last && tmp[k].off <= last->off + last->len + GAP_MERGE) {
        uint64_t end = tmp[k].off + tmp[k].len;
        if (end > last->off + last->len) last->len = end - last->off;
      } else {
        p->rspans = grow(p->rspans, &cap_r, p->nrspan + 1, sizeof *p->rspans);
        p->rspans[p->nrspan++] = tmp[k];
        b->nrspan++;
      }
    }
    for (uint32_t k = 0; k < b->nrspan; k++) b->read_bytes += p->rspans[b->rspan0 + k].len;
    b->msg0 = (uint32_t)p->nmsg;
    for (size_t k = i; k < j;) {
      size_t e = k;
      while (e < j && pc[e].rank == pc[k].rank) e++;
      p->msgs = grow(p->msgs, &cap_m, p->nmsg + 1, sizeof *p->msgs);
      pf_msg *m = &p->msgs[p->nmsg++];
      memset(m, 0, sizeof *m);
      m->rank = pc[k].rank;
      m->piece0 = (uint32_t)k;
      m->npiece = (uint32_t)(e - k);
      m->span0 = (uint32_t)p->nmspan;
      /* pieces of one rank are sorted by src_off: merge overlapping/adjacent ranges, place each piece */
      uint64_t pos_end = 0;
      for (size_t q = k; q < e; q++) {
        pf_span *last = m->nspan ? &p->mspans[p->nmspan - 1] : NULL;
        if (last && pc[q].src_off <= last->off + last->len) {
          uint64_t end = pc[q].src_off + pc[q].len;
          if (end > last->off + last->len) {
            pos_end += end - (last->off + last->len);
            last->len = end - last->off;
          }
          pc[q].msg_off = pos_end - (last->off + last->len - pc[q].src_off);
        } else {
          p->mspans = grow(p->mspans, &cap_ms, p->nmspan + 1, sizeof *p->mspans);
          p->mspans[p->nmspan].off = pc[q].src_off;
          p->mspans[p->nmspan].len = pc[q].len;
          p->nmspan++;
          m->nspan++;
          pc[q].msg_off = pos_end;
          pos_end += pc[q].len;
        }
      }
      m->bytes = pos_end;
      p->total_needed += pos_end;
      b->nmsg++;
      k = e;
    }
    p->total_read += b->read_bytes;
    p->nblock++;
    i = j;
  }
  free(tmp);

  /* reader assignment */
  free(p->load);
  p->load = calloc((size_t)p->ranks, sizeof *p->load);
  const uint64_t target = p->total_read / (uint64_t)p->ranks;
  const uint64_t limit = target + target / 100;
  uint32_t *order = malloc((p->nblock ? p->nblock : 1) * sizeof *order);
  size_t norder = 0;
  for (size_t i = 0; i < p->nblock; i++) p->blocks[i].reader = -1;
  if (assign == PF_ASSIGN_LOCAL)
    for (size_t i = 0; i < p->nblock; i++) {
      pf_block *b = &p->blocks[i];
      int r = p->msgs[b->msg0].rank;
      if (b->nmsg == 1 && p->load[r] + b->read_bytes <= limit) {
        b->reader = r;
        p->load[r] += b->read_bytes;
      }
    }
  /* the rest dealt in (file, index) order with a rotating start: consecutive blocks of one file go to different
   * readers and every reader gets blocks of every file */
  for (size_t i = 0; i < p->nblock; i++)
    if (p->blocks[i].reader < 0) order[norder++] = (uint32_t)i;
  for (size_t k = 0; k < norder; k++) {
    pf_block *b = &p->blocks[order[k]];
    int r = least_loaded(p->load, p->ranks, (int)(k % (size_t)p->ranks));
    b->reader = r;
    p->load[r] += b->read_bytes;
  }
  free(order);
  return 0;
}

size_t pf_queue(const pf_plan *p, int rank, int sequential, uint32_t *out) {
  key3 *keys = malloc((p->nblock ? p->nblock : 1) * sizeof *keys);
  uint64_t *ord = calloc(p->nsrc ? p->nsrc : 1, sizeof *ord);
  size_t n = 0;
  for (size_t i = 0; i < p->nblock; i++) {
    const pf_block *b = &p->blocks[i];
    if (b->reader != rank) continue;
    uint64_t k = ord[b->src]++; /* blocks are in (src, index) order, so this is the ordinal within the file */
    keys[n].a = (uint64_t)(int64_t)b->prio + (1ull << 31);
    keys[n].b = sequential ? (uint64_t)b->src << 32 | (uint32_t)k
                           : k * p->nsrc + (b->src + (uint32_t)rank) % p->nsrc;
    keys[n].i = (uint32_t)i;
    n++;
  }
  qsort(keys, n, sizeof *keys, cmp_key3);
  for (size_t i = 0; i < n; i++) out[i] = keys[i].i;
  free(keys);
  free(ord);
  return n;
}

const pf_msg *pf_block_msg(const pf_plan *p, const pf_block *b, int rank) {
  for (uint32_t k = 0; k < b->nmsg; k++)
    if (p->msgs[b->msg0 + k].rank == rank) return &p->msgs[b->msg0 + k];
  return NULL;
}

void pf_free(pf_plan *p) {
  for (uint32_t i = 0; i < p->nsrc; i++) free(p->srcs[i]);
  for (uint32_t i = 0; i < p->ngroup; i++) free(p->groups[i].name);
  for (uint32_t i = 0; i < p->ndest; i++) free(p->dests[i].path), free(p->dests[i].sidecar);
  for (uint32_t i = 0; i < p->nlink; i++) free(p->links[i].link), free(p->links[i].target);
  free(p->srcs), free(p->groups), free(p->dests), free(p->links), free(p->copies), free(p->tmpl_first), free(p->tmpl_count);
  free(p->pieces), free(p->blocks), free(p->rspans), free(p->mspans), free(p->msgs), free(p->load), free(p->src_hash);
  memset(p, 0, sizeof *p);
}
