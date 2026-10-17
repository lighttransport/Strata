/* ucfile - move or shard files between nodes over ucomm (InfiniBand verbs, or reliable UDP over Ethernet/IPoIB).
 *
 *   ucfile agent [--name N] [--port P] --root DIR [--root DIR...]   serve one node: discovery + control + jobs
 *   ucfile discover [--port P] [--timeout MS] [--json]              list agents on every broadcast domain
 *   ucfile run JOB.json [--dry-run]                                  plan, check and run a transfer job
 *   ucfile df|ls|stat|rm|mkdir NODE [PATH...] [--port P]             manage files on a node (inside its roots)
 *
 * NODE is an agent name (found by discovery) or host[:port]. Every node of a job runs one ucomm rank: the
 * coordinator (`run`) is rank 0 and sends the expanded job to the agents over their control port; all ranks then
 * walk the same item list (a sender and a receiver per item). Writes go to DEST.part with a resume state
 * DEST.part.ucstate; a finished file is fsynced and renamed. An agent reads and writes only under its --root
 * directories. Job file format: see README.md (ucfile). */
#define _GNU_SOURCE
#include "ucomm.h"
#include "ucomm_file.h"
#include "ujson.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define UCFILE_PORT 29610
#define UCFILE_VERSION "1"
#define MAX_NODES UCOMM_MAX_WORLD
#define MAX_ROOTS 16

static int g_port = UCFILE_PORT;
static char g_name[256];
static char *g_roots[MAX_ROOTS];
static int g_nroots;
static int g_verbose;
static volatile int g_busy;

static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "ucfile: ");
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  exit(1);
}
static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ local facts */

static int ib_active(void) {
  DIR *d = opendir("/sys/class/infiniband");
  if (!d) return 0;
  int active = 0;
  struct dirent *e;
  while (!active && (e = readdir(d))) {
    if (e->d_name[0] == '.') continue;
    for (int port = 1; port <= 2 && !active; port++) {
      char path[512], state[64] = {0};
      snprintf(path, sizeof path, "/sys/class/infiniband/%s/ports/%d/state", e->d_name, port);
      FILE *f = fopen(path, "r");
      if (!f) continue;
      if (fgets(state, sizeof state, f) && strstr(state, "ACTIVE")) active = 1;
      fclose(f);
    }
  }
  closedir(d);
  return active;
}
static int is_ib_ifname(const char *n) { return !strncmp(n, "ib", 2); }

/* IPv4 addresses with interface name and broadcast address; returns count. */
typedef struct { char ifname[IF_NAMESIZE]; char addr[INET_ADDRSTRLEN]; char bcast[INET_ADDRSTRLEN]; uint32_t ip, mask; } ifaddr4;
static int local_addrs(ifaddr4 *out, int cap) {
  struct ifaddrs *ifs, *i;
  int n = 0;
  if (getifaddrs(&ifs)) return 0;
  for (i = ifs; i && n < cap; i = i->ifa_next) {
    if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || (i->ifa_flags & IFF_LOOPBACK) || !(i->ifa_flags & IFF_UP)) continue;
    ifaddr4 *a = &out[n++];
    memset(a, 0, sizeof *a);
    snprintf(a->ifname, sizeof a->ifname, "%s", i->ifa_name);
    struct sockaddr_in *s = (struct sockaddr_in *)i->ifa_addr;
    inet_ntop(AF_INET, &s->sin_addr, a->addr, sizeof a->addr);
    a->ip = ntohl(s->sin_addr.s_addr);
    a->mask = i->ifa_netmask ? ntohl(((struct sockaddr_in *)i->ifa_netmask)->sin_addr.s_addr) : 0;
    if ((i->ifa_flags & IFF_BROADCAST) && i->ifa_broadaddr)
      inet_ntop(AF_INET, &((struct sockaddr_in *)i->ifa_broadaddr)->sin_addr, a->bcast, sizeof a->bcast);
  }
  freeifaddrs(ifs);
  return n;
}

/* ------------------------------------------------------------------ paths and roots */

/* Canonical form of a path whose last components may not exist yet (resolves the longest existing prefix). */
static int canon_path(const char *path, char *out) {
  char tmp[PATH_MAX];
  if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp || path[0] != '/') return -1;
  char rest[PATH_MAX] = "";
  for (;;) {
    char real[PATH_MAX];
    if (realpath(tmp, real)) {
      if (snprintf(out, PATH_MAX, "%s%s", real, rest) >= PATH_MAX) return -1;
      return 0;
    }
    char *slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) return -1;
    char next[PATH_MAX];
    if (snprintf(next, sizeof next, "%s%s", slash, rest) >= (int)sizeof next) return -1;
    if (strstr(slash, "/..") == slash) return -1; /* no '..' into unresolved parts */
    snprintf(rest, sizeof rest, "%s", next);
    *slash = 0;
  }
}
/* Agents only touch paths inside their roots; the coordinator's own paths are not restricted. */
static int path_allowed(const char *path, char *canon) {
  if (canon_path(path, canon)) return 0;
  if (!g_nroots) return 1;
  for (int i = 0; i < g_nroots; i++) {
    size_t n = strlen(g_roots[i]);
    if (!strncmp(canon, g_roots[i], n) && (canon[n] == 0 || canon[n] == '/' || g_roots[i][n - 1] == '/')) return 1;
  }
  return 0;
}
static int mkdir_p(const char *dir) {
  char tmp[PATH_MAX];
  snprintf(tmp, sizeof tmp, "%s", dir);
  for (char *p = tmp + 1; *p; p++)
    if (*p == '/') {
      *p = 0;
      if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
      *p = '/';
    }
  return mkdir(tmp, 0755) && errno != EEXIST ? -1 : 0;
}
static void parent_dir(const char *path, char *out) {
  snprintf(out, PATH_MAX, "%s", path);
  char *s = strrchr(out, '/');
  if (s && s != out) *s = 0;
  else snprintf(out, PATH_MAX, "/");
}
/* Free bytes on the filesystem that holds path (or its nearest existing parent). */
static int fs_free(const char *path, uint64_t *free_bytes, uint64_t *dev) {
  char p[PATH_MAX];
  snprintf(p, sizeof p, "%s", path);
  struct stat st;
  while (stat(p, &st)) {
    char *s = strrchr(p, '/');
    if (!s) return -1;
    if (s == p) { p[1] = 0; break; }
    *s = 0;
  }
  struct statvfs v;
  if (statvfs(p, &v)) return -1;
  *free_bytes = (uint64_t)v.f_bavail * v.f_frsize;
  if (dev) *dev = stat(p, &st) ? 0 : (uint64_t)st.st_dev;
  return 0;
}

/* ------------------------------------------------------------------ line-based control connections */

static int send_all(int fd, const char *p, size_t n) {
  while (n) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k;
    n -= (size_t)k;
  }
  return 0;
}
static int send_line(int fd, uj_buf *b) {
  uj_put(b, "\n");
  return send_all(fd, b->s, b->n);
}
/* Reads one '\n'-terminated line (up to 64 MiB); caller frees. */
static char *recv_line(int fd, int timeout_ms) {
  size_t cap = 4096, n = 0;
  char *s = malloc(cap);
  for (;;) {
    struct pollfd p = {fd, POLLIN, 0};
    if (timeout_ms > 0 && poll(&p, 1, timeout_ms) <= 0) { free(s); return NULL; }
    if (n + 1 >= cap) {
      if (cap >= (64u << 20)) { free(s); return NULL; }
      s = realloc(s, cap *= 2);
    }
    ssize_t k = recv(fd, s + n, 1, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) { free(s); return NULL; }
    if (s[n] == '\n') { s[n] = 0; return s; }
    n++;
  }
}
static int tcp_connect(const char *host, int port, int timeout_ms) {
  char ps[16];
  snprintf(ps, sizeof ps, "%d", port);
  struct addrinfo hints = {0}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host, ps, &hints, &res)) return -1;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  fcntl(fd, F_SETFL, O_NONBLOCK);
  int r = connect(fd, res->ai_addr, res->ai_addrlen);
  freeaddrinfo(res);
  if (r && errno == EINPROGRESS) {
    struct pollfd p = {fd, POLLOUT, 0};
    int err = 0;
    socklen_t len = sizeof err;
    if (poll(&p, 1, timeout_ms) == 1 && !getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) && !err) r = 0;
  }
  if (r) { close(fd); return -1; }
  fcntl(fd, F_SETFL, 0);
  return fd;
}
/* One request, one final response line; progress lines (`"progress":...`) are printed when verbose. */
static uj_node *request(const char *host, int port, uj_buf *req, int timeout_ms) {
  int fd = tcp_connect(host, port, 5000);
  if (fd < 0) return NULL;
  uj_node *reply = NULL;
  if (!send_line(fd, req)) {
    for (;;) {
      char *line = recv_line(fd, timeout_ms);
      if (!line) break;
      uj_node *v = uj_parse(line, strlen(line), NULL);
      if (v && uj_get(v, "progress")) {
        if (g_verbose) fprintf(stderr, "%s\n", line);
        uj_free(v);
        free(line);
        continue;
      }
      free(line);
      reply = v;
      break;
    }
  }
  close(fd);
  return reply;
}

/* ------------------------------------------------------------------ discovery */

static void info_json(uj_buf *b) {
  ifaddr4 a[32];
  int n = local_addrs(a, 32);
  uj_put(b, "{\"ucfile\":\"" UCFILE_VERSION "\",\"name\":");
  uj_put_str(b, g_name);
  uj_put(b, ",\"port\":%d,\"ib\":%s,\"busy\":%s,\"addrs\":[", g_port, ib_active() ? "true" : "false", g_busy ? "true" : "false");
  for (int i = 0; i < n; i++) {
    uj_put(b, "%s{\"if\":", i ? "," : "");
    uj_put_str(b, a[i].ifname);
    uj_put(b, ",\"addr\":");
    uj_put_str(b, a[i].addr);
    uj_put(b, "}");
  }
  uj_put(b, "],\"roots\":[");
  for (int i = 0; i < g_nroots; i++) {
    uint64_t fr = 0;
    fs_free(g_roots[i], &fr, NULL);
    uj_put(b, "%s{\"path\":", i ? "," : "");
    uj_put_str(b, g_roots[i]);
    uj_put(b, ",\"free\":%llu}", (unsigned long long)fr);
  }
  uj_put(b, "]}");
}

static void *discovery_thread(void *arg) {
  (void)arg;
  int fd = socket(AF_INET, SOCK_DGRAM, 0), one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)g_port);
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, (struct sockaddr *)&sa, sizeof sa)) { perror("ucfile agent: discovery bind"); return NULL; }
  char buf[512];
  for (;;) {
    struct sockaddr_in from;
    socklen_t fl = sizeof from;
    ssize_t k = recvfrom(fd, buf, sizeof buf - 1, 0, (struct sockaddr *)&from, &fl);
    if (k < 7 || memcmp(buf, "UCFILE?", 7)) continue;
    uj_buf b = {0};
    uj_put(&b, "UCFILE!");
    info_json(&b);
    sendto(fd, b.s, b.n, 0, (struct sockaddr *)&from, fl);
    free(b.s);
  }
  return NULL;
}

typedef struct {
  char name[256];
  char addr[INET_ADDRSTRLEN];   /* address the reply came from (reachable from here) */
  char local[INET_ADDRSTRLEN];  /* our address on that subnet */
  int port, ib, ipoib;          /* ipoib: reached over an IPoIB interface */
  uj_node *info;
} peer_t;

/* Broadcasts a probe on every interface (and to `extra` unicast/broadcast addresses), collects replies. */
static int discover(int port, int timeout_ms, const char **extra, int nextra, peer_t *out, int cap) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0), one = 1;
  setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);
  ifaddr4 a[32];
  int na = local_addrs(a, 32), n = 0;
  const char *probe = "UCFILE?";
  struct sockaddr_in to = {0};
  to.sin_family = AF_INET;
  to.sin_port = htons((uint16_t)port);
  for (int i = 0; i < na + nextra + 1; i++) {
    /* interface broadcast addresses, then the extra targets (addr or addr:port), then the limited broadcast */
    char dst[64];
    snprintf(dst, sizeof dst, "%s", i < na ? a[i].bcast : i < na + nextra ? extra[i - na] : "255.255.255.255");
    char *colon = strchr(dst, ':');
    to.sin_port = htons((uint16_t)(colon ? atoi(colon + 1) : port));
    if (colon) *colon = 0;
    if (!dst[0] || inet_pton(AF_INET, dst, &to.sin_addr) != 1) continue;
    sendto(fd, probe, strlen(probe), 0, (struct sockaddr *)&to, sizeof to);
  }
  const double end = now_s() + timeout_ms / 1000.0;
  char buf[65536];
  while (now_s() < end && n < cap) {
    struct pollfd p = {fd, POLLIN, 0};
    if (poll(&p, 1, (int)((end - now_s()) * 1000) + 1) <= 0) break;
    struct sockaddr_in from;
    socklen_t fl = sizeof from;
    ssize_t k = recvfrom(fd, buf, sizeof buf - 1, 0, (struct sockaddr *)&from, &fl);
    if (k < 7 || memcmp(buf, "UCFILE!", 7)) continue;
    buf[k] = 0;
    uj_node *info = uj_parse(buf + 7, (size_t)k - 7, NULL);
    if (!info) continue;
    peer_t pr = {0};
    snprintf(pr.name, sizeof pr.name, "%s", uj_str(info, "name", "?"));
    inet_ntop(AF_INET, &from.sin_addr, pr.addr, sizeof pr.addr);
    pr.port = (int)uj_num(info, "port", port);
    pr.ib = uj_bool(info, "ib", 0);
    const uint32_t ip = ntohl(from.sin_addr.s_addr);
    for (int i = 0; i < na; i++)
      if (a[i].mask && (a[i].ip & a[i].mask) == (ip & a[i].mask)) {
        snprintf(pr.local, sizeof pr.local, "%s", a[i].addr);
        pr.ipoib = is_ib_ifname(a[i].ifname);
      }
    pr.info = info;
    /* one entry per name: prefer the reply that came over IPoIB */
    int dup = -1;
    for (int i = 0; i < n; i++)
      if (!strcmp(out[i].name, pr.name)) dup = i;
    if (dup < 0) out[n++] = pr;
    else if (pr.ipoib && !out[dup].ipoib) { uj_free(out[dup].info); out[dup] = pr; }
    else uj_free(info);
  }
  close(fd);
  return n;
}

/* The local address the kernel would use to reach host (a connected UDP socket sends nothing). */
static int local_route(const char *host, char *out) {
  struct addrinfo hints = {0}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  if (getaddrinfo(host, "9", &hints, &res)) return -1;
  int fd = socket(AF_INET, SOCK_DGRAM, 0), r = -1;
  if (fd >= 0 && !connect(fd, res->ai_addr, res->ai_addrlen)) {
    struct sockaddr_in sa;
    socklen_t len = sizeof sa;
    if (!getsockname(fd, (struct sockaddr *)&sa, &len) && inet_ntop(AF_INET, &sa.sin_addr, out, INET_ADDRSTRLEN)) r = 0;
  }
  if (fd >= 0) close(fd);
  freeaddrinfo(res);
  return r;
}

/* host[:port] or an agent name (by discovery) -> host and port */
static int resolve_node(const char *node, char *host, int *port) {
  const char *colon = strrchr(node, ':');
  if (strchr(node, '.') || colon) {
    snprintf(host, INET_ADDRSTRLEN + 64, "%.*s", colon ? (int)(colon - node) : (int)strlen(node), node);
    *port = colon ? atoi(colon + 1) : g_port;
    return 0;
  }
  peer_t peers[32];
  int n = discover(g_port, 700, NULL, 0, peers, 32), found = -1;
  for (int i = 0; i < n; i++)
    if (!strcmp(peers[i].name, node)) found = i;
  if (found >= 0) { snprintf(host, INET_ADDRSTRLEN, "%s", peers[found].addr); *port = peers[found].port; }
  for (int i = 0; i < n; i++) uj_free(peers[i].info);
  return found >= 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ file operations (agent side and local) */

static void stat_json(uj_buf *b, const char *path) {
  struct stat st;
  char canon[PATH_MAX];
  if (!path_allowed(path, canon)) { uj_put(b, "{\"error\":\"outside the agent's roots\"}"); return; }
  uint64_t fr = 0, dev = 0;
  fs_free(canon, &fr, &dev);
  if (stat(canon, &st)) {
    uj_put(b, "{\"exists\":false,\"free\":%llu,\"dev\":%llu}", (unsigned long long)fr, (unsigned long long)dev);
    return;
  }
  uj_put(b, "{\"exists\":true,\"dir\":%s,\"size\":%llu,\"mtime\":%lld,\"free\":%llu,\"dev\":%llu",
         S_ISDIR(st.st_mode) ? "true" : "false", (unsigned long long)st.st_size, (long long)st.st_mtime,
         (unsigned long long)fr, (unsigned long long)dev);
  char part[PATH_MAX + 8];
  snprintf(part, sizeof part, "%s.part", canon);
  if (!stat(part, &st)) uj_put(b, ",\"part\":%llu", (unsigned long long)st.st_size);
  uj_put(b, "}");
}
static void op_stat_any(uj_buf *b, const char *path) {
  /* stat that also reports a resumable .part when the file itself does not exist */
  struct stat st;
  char canon[PATH_MAX], part[PATH_MAX + 8];
  if (path_allowed(path, canon) && stat(canon, &st)) {
    snprintf(part, sizeof part, "%s.part", canon);
    uint64_t fr = 0, dev = 0;
    fs_free(canon, &fr, &dev);
    if (!stat(part, &st)) {
      uj_put(b, "{\"exists\":false,\"part\":%llu,\"free\":%llu,\"dev\":%llu}", (unsigned long long)st.st_size,
             (unsigned long long)fr, (unsigned long long)dev);
      return;
    }
  }
  stat_json(b, path);
}
static void op_ls(uj_buf *b, const char *path) {
  char canon[PATH_MAX];
  if (!path_allowed(path, canon)) { uj_put(b, "{\"error\":\"outside the agent's roots\"}"); return; }
  DIR *d = opendir(canon);
  if (!d) { uj_put(b, "{\"error\":\"cannot open directory\"}"); return; }
  uj_put(b, "{\"path\":");
  uj_put_str(b, canon);
  uj_put(b, ",\"entries\":[");
  struct dirent *e;
  int first = 1;
  while ((e = readdir(d))) {
    if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
    char full[PATH_MAX * 2];
    snprintf(full, sizeof full, "%s/%s", canon, e->d_name);
    struct stat st;
    if (lstat(full, &st)) continue;
    uj_put(b, "%s{\"name\":", first ? "" : ",");
    uj_put_str(b, e->d_name);
    uj_put(b, ",\"dir\":%s,\"size\":%llu,\"mtime\":%lld}", S_ISDIR(st.st_mode) ? "true" : "false",
           (unsigned long long)st.st_size, (long long)st.st_mtime);
    first = 0;
  }
  closedir(d);
  uj_put(b, "]}");
}
static void op_rm(uj_buf *b, const char *path) {
  char canon[PATH_MAX];
  if (!path_allowed(path, canon)) { uj_put(b, "{\"error\":\"outside the agent's roots\"}"); return; }
  for (int i = 0; i < g_nroots; i++)
    if (!strcmp(canon, g_roots[i])) { uj_put(b, "{\"error\":\"refusing to remove a root\"}"); return; }
  struct stat st;
  if (lstat(canon, &st)) { uj_put(b, "{\"error\":\"no such file\"}"); return; }
  /* files, or empty directories; also a resume state next to a .part */
  int r = S_ISDIR(st.st_mode) ? rmdir(canon) : unlink(canon);
  if (r) { uj_put(b, "{\"error\":"); uj_put_str(b, strerror(errno)); uj_put(b, "}"); return; }
  if (strlen(canon) > 5 && !strcmp(canon + strlen(canon) - 5, ".part")) {
    char state[PATH_MAX + 16];
    snprintf(state, sizeof state, "%s.ucstate", canon);
    unlink(state);
  }
  uj_put(b, "{\"removed\":");
  uj_put_str(b, canon);
  uj_put(b, "}");
}
static void op_mkdir(uj_buf *b, const char *path) {
  char canon[PATH_MAX];
  if (!path_allowed(path, canon)) { uj_put(b, "{\"error\":\"outside the agent's roots\"}"); return; }
  if (mkdir_p(canon)) { uj_put(b, "{\"error\":"); uj_put_str(b, strerror(errno)); uj_put(b, "}"); return; }
  uj_put(b, "{\"created\":");
  uj_put_str(b, canon);
  uj_put(b, "}");
}

/* ------------------------------------------------------------------ job execution (every rank) */

typedef struct {
  char src_node[64], dst_node[64];
  char src_path[PATH_MAX], dst_path[PATH_MAX];
  uint64_t offset, length, src_size;
  int64_t src_mtime;
} item_t;

typedef struct {
  int world, nitems;
  char root[128];
  char nodes[MAX_NODES][64];
  ucomm_backend backend;
  ucomm_file_opts opt;
  int overwrite, keep_mtime, timeout_ms;
  item_t *items;
} job_t;

static int node_rank(const job_t *j, const char *name) {
  for (int r = 0; r < j->world; r++)
    if (!strcmp(j->nodes[r], name)) return r;
  return -1;
}

/* resume state next to DEST.part: magic, length, chunk, source size/mtime/offset, then the chunk bitmap */
typedef struct { char magic[8]; uint64_t length, chunk, src_size, src_offset; int64_t src_mtime; } state_hdr;

typedef struct {
  int fd, state_fd;
  uint8_t *bitmap;
  size_t bitmap_bytes;
  state_hdr hdr;
  uint64_t since_sync;
  int progress_fd; /* control connection for progress lines, or -1 */
  const item_t *it;
  int index;
  double t0;
} recv_ctx;

static void persist_state(recv_ctx *r) {
  fdatasync(r->fd);
  if (pwrite(r->state_fd, &r->hdr, sizeof r->hdr, 0) != (ssize_t)sizeof r->hdr ||
      pwrite(r->state_fd, r->bitmap, r->bitmap_bytes, sizeof r->hdr) != (ssize_t)r->bitmap_bytes)
    fprintf(stderr, "ucfile: cannot write the resume state\n");
  fdatasync(r->state_fd);
}
static void on_chunk(void *user, uint64_t chunk, uint64_t done) {
  (void)chunk;
  recv_ctx *r = user;
  if (++r->since_sync >= 32) { persist_state(r); r->since_sync = 0; }
  static double last;
  const double t = now_s();
  if (t - last >= 2.0 || done == r->it->length) {
    last = t;
    uj_buf b = {0};
    uj_put(&b, "{\"progress\":%d,\"bytes\":%llu,\"of\":%llu,\"elapsed\":%.1f}", r->index, (unsigned long long)done,
           (unsigned long long)r->it->length, t - r->t0);
    if (r->progress_fd >= 0) send_line(r->progress_fd, &b);
    else if (g_verbose) fprintf(stderr, "%s", b.s);
    free(b.s);
  }
}

static void result_line(uj_buf *b, int index, const char *role, const char *status, const ucomm_file_stats *st) {
  if (b->n > 1) uj_put(b, ",");
  uj_put(b, "{\"item\":%d,\"role\":\"%s\",\"status\":", index, role);
  uj_put_str(b, status);
  if (st)
    uj_put(b, ",\"bytes\":%llu,\"chunks\":%llu,\"skipped\":%llu,\"resent\":%llu,\"seconds\":%.3f,\"GBps\":%.3f",
           (unsigned long long)st->bytes, (unsigned long long)st->chunks, (unsigned long long)st->skipped,
           (unsigned long long)st->resent, st->seconds, st->seconds > 0 ? st->bytes / st->seconds / 1e9 : 0.0);
  uj_put(b, "}");
}

static const char *receive_item(ucomm_t *c, const job_t *j, int i, int peer, int progress_fd, ucomm_file_stats *st) {
  const item_t *it = &j->items[i];
  char canon[PATH_MAX], dir[PATH_MAX], part[PATH_MAX + 8], state[PATH_MAX + 16];
  if (!path_allowed(it->dst_path, canon)) return "destination outside the agent's roots";
  struct stat sb;
  if (!stat(canon, &sb) && !j->overwrite) return "destination exists (set \"overwrite\": true)";
  parent_dir(canon, dir);
  if (mkdir_p(dir)) return "cannot create the destination directory";
  snprintf(part, sizeof part, "%s.part", canon);
  snprintf(state, sizeof state, "%s.ucstate", part);
  recv_ctx r = {0};
  r.it = it;
  r.index = i;
  r.progress_fd = progress_fd;
  r.t0 = now_s();
  r.bitmap_bytes = ucomm_file_bitmap_bytes(it->length, &j->opt);
  r.bitmap = calloc(r.bitmap_bytes ? r.bitmap_bytes : 1, 1);
  memcpy(r.hdr.magic, "UCFS1\0\0\0", 8);
  r.hdr.length = it->length;
  r.hdr.chunk = j->opt.chunk_bytes;
  r.hdr.src_size = it->src_size;
  r.hdr.src_offset = it->offset;
  r.hdr.src_mtime = it->src_mtime;
  r.fd = open(part, O_RDWR | O_CREAT, 0644);
  r.state_fd = open(state, O_RDWR | O_CREAT, 0644);
  if (r.fd < 0 || r.state_fd < 0) { free(r.bitmap); return "cannot open the destination"; }
  /* resume only a state written for the same source range and chunking */
  state_hdr old;
  int resumed = pread(r.state_fd, &old, sizeof old, 0) == (ssize_t)sizeof old && !memcmp(&old, &r.hdr, sizeof old) &&
                pread(r.state_fd, r.bitmap, r.bitmap_bytes, sizeof old) == (ssize_t)r.bitmap_bytes;
  if (!resumed) {
    memset(r.bitmap, 0, r.bitmap_bytes);
    if (ftruncate(r.fd, 0) || (it->length && posix_fallocate(r.fd, 0, (off_t)it->length))) {
      close(r.fd); close(r.state_fd); free(r.bitmap);
      return "cannot allocate the destination (disk full?)";
    }
    persist_state(&r);
  }
  ucomm_file_opts o = j->opt;
  o.on_chunk = on_chunk;
  o.user = &r;
  ucomm_status s = ucomm_file_recv(c, peer, (uint32_t)(2 * i), r.fd, 0, it->length, &o, r.bitmap, st);
  persist_state(&r);
  const char *status = "ok";
  if (s != UCOMM_OK) status = ucomm_strerror(s);
  else if (fsync(r.fd)) status = "fsync failed";
  else if (rename(part, canon)) status = "rename failed";
  else {
    unlink(state);
    if (j->keep_mtime && it->src_mtime) {
      struct timespec ts[2] = {{0, UTIME_OMIT}, {(time_t)it->src_mtime, 0}};
      futimens(r.fd, ts);
    }
    int dfd = open(dir, O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) { fsync(dfd); close(dfd); }
  }
  close(r.fd);
  close(r.state_fd);
  free(r.bitmap);
  return status;
}

static const char *local_copy(const job_t *j, int i, ucomm_file_stats *st) {
  const item_t *it = &j->items[i];
  char sc[PATH_MAX], dc[PATH_MAX];
  if (!path_allowed(it->src_path, sc) || !path_allowed(it->dst_path, dc)) return "path outside the agent's roots";
  struct stat sb;
  if (!stat(dc, &sb) && !j->overwrite) return "destination exists (set \"overwrite\": true)";
  char dir[PATH_MAX], part[PATH_MAX + 8];
  parent_dir(dc, dir);
  if (mkdir_p(dir)) return "cannot create the destination directory";
  snprintf(part, sizeof part, "%s.part", dc);
  int in = open(sc, O_RDONLY), out = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (in < 0 || out < 0) return "cannot open";
  const double t0 = now_s();
  loff_t off_in = (loff_t)it->offset, off_out = 0;
  uint64_t left = it->length;
  while (left) {
    ssize_t k = copy_file_range(in, &off_in, out, &off_out, left > (1u << 30) ? (1u << 30) : left, 0);
    if (k <= 0) break;
    left -= (uint64_t)k;
  }
  const char *status = left ? "copy failed" : fsync(out) ? "fsync failed" : rename(part, dc) ? "rename failed" : "ok";
  close(in);
  close(out);
  st->bytes = it->length - left;
  st->chunks = 1;
  st->seconds = now_s() - t0;
  return status;
}

/* Runs this rank's part of the job; appends one result object per item it took part in. */
static int run_job(const job_t *j, int rank, int progress_fd, uj_buf *results) {
  ucomm_config cfg;
  ucomm_config_default(&cfg);
  cfg.rank = rank;
  cfg.world = j->world;
  cfg.root_addr = j->root;
  cfg.backend = j->backend;
  cfg.timeout_ms = j->timeout_ms;
  cfg.verbose = g_verbose;
  ucomm_t *c = NULL;
  int failures = 0;
  uj_put(results, "[");
  if (j->world > 1) {
    ucomm_status s = ucomm_init(&cfg, &c);
    if (s != UCOMM_OK) {
      uj_put(results, "{\"error\":");
      uj_put_str(results, ucomm_strerror(s));
      uj_put(results, "}]");
      return 1;
    }
    if (g_verbose && rank == 0) fprintf(stderr, "ucfile: %d ranks, backend %s\n", j->world, ucomm_backend_name(c));
  }
  for (int i = 0; i < j->nitems; i++) {
    const item_t *it = &j->items[i];
    const int src = node_rank(j, it->src_node), dst = node_rank(j, it->dst_node);
    ucomm_file_stats st = {0};
    const char *status = NULL, *role = NULL;
    if (src == rank && dst == rank) { role = "copy"; status = local_copy(j, i, &st); }
    else if (src == rank) {
      role = "send";
      char canon[PATH_MAX];
      int fd = path_allowed(it->src_path, canon) ? open(canon, O_RDONLY) : -1;
      if (fd < 0) status = "cannot open the source";
      else {
        posix_fadvise(fd, (off_t)it->offset, (off_t)it->length, POSIX_FADV_SEQUENTIAL);
        ucomm_status s = ucomm_file_send(c, dst, (uint32_t)(2 * i), fd, it->offset, it->length, &j->opt, &st);
        status = s == UCOMM_OK ? "ok" : ucomm_strerror(s);
        close(fd);
      }
    } else if (dst == rank) { role = "recv"; status = receive_item(c, j, i, src, progress_fd, &st); }
    if (!role) continue;
    if (strcmp(status, "ok")) failures++;
    result_line(results, i, role, status, &st);
    /* a side that failed before the transfer started leaves its peer waiting: stop rather than hang on */
    if (strcmp(status, "ok") && st.bytes == 0 && strcmp(role, "copy")) break;
  }
  uj_put(results, "]");
  if (c) ucomm_finalize(c);
  return failures;
}

static int parse_job(const uj_node *v, job_t *j, char *err, size_t errlen) {
  memset(j, 0, sizeof *j);
  ucomm_file_opts_default(&j->opt);
  j->world = (int)uj_num(v, "world", 0);
  snprintf(j->root, sizeof j->root, "%s", uj_str(v, "root", ""));
  const char *be = uj_str(v, "backend", "auto");
  j->backend = !strcmp(be, "ib") ? UCOMM_BACKEND_IB : !strcmp(be, "udp") ? UCOMM_BACKEND_UDP : UCOMM_BACKEND_AUTO;
  j->opt.chunk_bytes = (size_t)uj_num(v, "chunk_bytes", 16 << 20);
  j->opt.depth = (int)uj_num(v, "depth", 4);
  j->opt.verify = !strcmp(uj_str(v, "verify", "none"), "crc32c") ? UCOMM_FILE_VERIFY_CRC32C : UCOMM_FILE_VERIFY_NONE;
  j->overwrite = uj_bool(v, "overwrite", 0);
  j->keep_mtime = uj_bool(v, "keep_mtime", 1);
  j->timeout_ms = (int)uj_num(v, "timeout_ms", 120000);
  const uj_node *nodes = uj_get(v, "ranks"), *items = uj_get(v, "items");
  if (j->world < 1 || j->world > MAX_NODES || !nodes || uj_len(nodes) != j->world || !items ||
      j->opt.chunk_bytes % 4096 || !j->opt.chunk_bytes) {
    snprintf(err, errlen, "invalid expanded job");
    return -1;
  }
  int r = 0;
  for (const uj_node *n = nodes->child; n; n = n->next) snprintf(j->nodes[r++], 64, "%s", n->str ? n->str : "");
  j->nitems = uj_len(items);
  j->items = calloc((size_t)j->nitems + 1, sizeof *j->items);
  int i = 0;
  for (const uj_node *n = items->child; n; n = n->next, i++) {
    item_t *it = &j->items[i];
    snprintf(it->src_node, sizeof it->src_node, "%s", uj_str(n, "src_node", ""));
    snprintf(it->dst_node, sizeof it->dst_node, "%s", uj_str(n, "dst_node", ""));
    snprintf(it->src_path, sizeof it->src_path, "%s", uj_str(n, "src_path", ""));
    snprintf(it->dst_path, sizeof it->dst_path, "%s", uj_str(n, "dst_path", ""));
    it->offset = (uint64_t)uj_num(n, "offset", 0);
    it->length = (uint64_t)uj_num(n, "length", 0);
    it->src_size = (uint64_t)uj_num(n, "src_size", 0);
    it->src_mtime = (int64_t)uj_num(n, "src_mtime", 0);
  }
  return 0;
}

/* ------------------------------------------------------------------ agent */

static void handle_client(int fd) {
  char *line = recv_line(fd, 30000);
  if (!line) return;
  const char *err = NULL;
  uj_node *req = uj_parse(line, strlen(line), &err);
  free(line);
  uj_buf b = {0};
  const char *op = uj_str(req, "op", "");
  if (!req) uj_put(&b, "{\"error\":\"bad request\"}");
  else if (!strcmp(op, "info")) info_json(&b);
  else if (!strcmp(op, "stat")) op_stat_any(&b, uj_str(req, "path", ""));
  else if (!strcmp(op, "ls")) op_ls(&b, uj_str(req, "path", ""));
  else if (!strcmp(op, "rm")) op_rm(&b, uj_str(req, "path", ""));
  else if (!strcmp(op, "mkdir")) op_mkdir(&b, uj_str(req, "path", ""));
  else if (!strcmp(op, "job")) {
    job_t j;
    char perr[128];
    const int rank = (int)uj_num(req, "rank", -1);
    if (g_busy) uj_put(&b, "{\"error\":\"busy with another job\"}");
    else if (parse_job(uj_get(req, "job"), &j, perr, sizeof perr) || rank < 1 || rank >= j.world) uj_put(&b, "{\"error\":\"invalid job\"}");
    else {
      /* Each job runs in its own process (ucomm allows one context per process). The agent watches the control
         connection: if the coordinator goes away, the job is killed; its resume state is on disk. */
      g_busy = 1;
      uj_buf ack = {0};
      uj_put(&ack, "{\"progress\":-1,\"accepted\":true}");
      send_line(fd, &ack);
      free(ack.s);
      fflush(NULL);
      pid_t child = fork();
      if (child == 0) {
        uj_buf res = {0}, out = {0};
        const double t0 = now_s();
        int failures = run_job(&j, rank, fd, &res);
        uj_put(&out, "{\"node\":");
        uj_put_str(&out, g_name);
        uj_put(&out, ",\"failures\":%d,\"seconds\":%.3f,\"results\":%s}", failures, now_s() - t0, res.s);
        send_line(fd, &out);
        fflush(NULL);
        _exit(failures ? 1 : 0);
      }
      int status = 0, killed = 0;
      while (child > 0 && waitpid(child, &status, WNOHANG) == 0) {
        struct pollfd p = {fd, POLLRDHUP, 0};
        if (poll(&p, 1, 200) > 0 && (p.revents & (POLLRDHUP | POLLHUP | POLLERR))) {
          kill(child, SIGKILL);
          waitpid(child, &status, 0);
          killed = 1;
          fprintf(stderr, "ucfile agent: the coordinator went away; job stopped (resume state kept)\n");
        }
      }
      free(j.items);
      g_busy = 0;
      uj_free(req);
      (void)killed;
      return; /* the child sent the final line */
    }
  } else uj_put(&b, "{\"error\":\"unknown op\"}");
  send_line(fd, &b);
  free(b.s);
  uj_free(req);
}

static int cmd_agent(void) {
  if (!g_nroots) die("agent: give at least one --root DIR (the directories this node serves)");
  for (int i = 0; i < g_nroots; i++) {
    char canon[PATH_MAX];
    if (!realpath(g_roots[i], canon)) die("agent: root %s does not exist", g_roots[i]);
    g_roots[i] = strdup(canon);
  }
  signal(SIGPIPE, SIG_IGN);
  pthread_t t;
  pthread_create(&t, NULL, discovery_thread, NULL);
  int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)g_port);
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 8)) die("agent: cannot listen on port %d: %s", g_port, strerror(errno));
  fprintf(stderr, "ucfile agent %s: port %d (udp discovery, tcp control), ib %s, roots:", g_name, g_port,
          ib_active() ? "active" : "none");
  for (int i = 0; i < g_nroots; i++) fprintf(stderr, " %s", g_roots[i]);
  fputc('\n', stderr);
  for (;;) {
    int fd = accept(ls, NULL, NULL);
    if (fd < 0) continue;
    handle_client(fd); /* one at a time; discovery keeps answering (busy) meanwhile */
    close(fd);
  }
  return 0;
}

/* ------------------------------------------------------------------ coordinator */

typedef struct { char name[64]; char host[INET_ADDRSTRLEN + 64]; char local[INET_ADDRSTRLEN]; int port, self, ipoib; } node_t;

static int find_node(node_t *nodes, int n, const char *name) {
  for (int i = 0; i < n; i++)
    if (!strcmp(nodes[i].name, name)) return i;
  return -1;
}
/* "node:/path" -> node, path */
static int split_spec(const char *spec, char *node, char *path) {
  const char *c = strchr(spec, ':');
  if (!c || c == spec || c[1] != '/') return -1;
  snprintf(node, 64, "%.*s", (int)(c - spec), spec);
  snprintf(path, PATH_MAX, "%s", c + 1);
  return 0;
}
static uj_node *node_op(const node_t *nd, const char *op, const char *path) {
  uj_buf b = {0};
  uj_put(&b, "{\"op\":\"%s\",\"path\":", op);
  uj_put_str(&b, path);
  uj_put(&b, "}");
  uj_node *r = NULL;
  if (nd->self) {
    uj_buf out = {0};
    if (!strcmp(op, "stat")) op_stat_any(&out, path);
    r = uj_parse(out.s, out.n, NULL);
    free(out.s);
  } else r = request(nd->host, nd->port, &b, 15000);
  free(b.s);
  return r;
}

static int cmd_run(const char *path, int dry_run) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *text = malloc((size_t)len + 1);
  if (fread(text, 1, (size_t)len, f) != (size_t)len) die("cannot read %s", path);
  fclose(f);
  const char *err = NULL;
  uj_node *cfg = uj_parse(text, (size_t)len, &err);
  free(text);
  if (!cfg) die("%s: %s", path, err ? err : "invalid JSON");
  if (uj_get(cfg, "port")) g_port = (int)uj_num(cfg, "port", g_port);
  g_verbose |= uj_bool(cfg, "verbose", 0);

  /* nodes: "self" plus the named agents ("addr": host, host:port, or "auto" for discovery) */
  node_t nodes[MAX_NODES];
  int nn = 0;
  const char *self = uj_str(cfg, "self", g_name);
  const uj_node *cn = uj_get(cfg, "nodes");
  peer_t peers[32];
  int np = -1;
  for (const uj_node *n = cn ? cn->child : NULL; n; n = n->next) {
    if (nn == MAX_NODES) die("at most %d nodes", MAX_NODES);
    node_t *nd = &nodes[nn++];
    memset(nd, 0, sizeof *nd);
    snprintf(nd->name, sizeof nd->name, "%s", n->key);
    nd->self = !strcmp(n->key, self);
    nd->port = (int)uj_num(n, "port", g_port);
    const char *addr = uj_str(n, "addr", "auto");
    if (nd->self) continue;
    if (strcmp(addr, "auto")) {
      const char *colon = strrchr(addr, ':');
      snprintf(nd->host, sizeof nd->host, "%.*s", colon ? (int)(colon - addr) : (int)strlen(addr), addr);
      if (colon) nd->port = atoi(colon + 1);
      continue;
    }
    if (np < 0) {
      const uj_node *ex = uj_get(cfg, "discover");
      const char *extra[16];
      int ne = 0;
      for (const uj_node *e = ex ? ex->child : NULL; e && ne < 16; e = e->next) extra[ne++] = e->str;
      np = discover(g_port, (int)uj_num(cfg, "discover_ms", 1000), extra, ne, peers, 32);
    }
    int found = -1;
    for (int i = 0; i < np; i++)
      if (!strcmp(peers[i].name, nd->name)) found = i;
    if (found < 0) die("node %s: no agent answered discovery (start `ucfile agent` there, or give \"addr\")", nd->name);
    snprintf(nd->host, sizeof nd->host, "%s", peers[found].addr);
    snprintf(nd->local, sizeof nd->local, "%s", peers[found].local);
    nd->port = peers[found].port;
    nd->ipoib = peers[found].ipoib;
  }
  if (find_node(nodes, nn, self) < 0) die("\"self\" (%s) is not among the nodes", self);

  /* items: whole files, byte ranges, or shards split by weights across several destinations */
  const uj_node *tr = uj_get(cfg, "transfers");
  item_t *items = calloc(4096, sizeof *items);
  int ni = 0;
  uj_buf manifest = {0};
  int nmanifest = 0;
  for (const uj_node *t = tr ? tr->child : NULL; t; t = t->next) {
    char sn[64], sp[PATH_MAX];
    if (split_spec(uj_str(t, "src", ""), sn, sp)) die("transfer: \"src\" must be node:/absolute/path");
    const int si = find_node(nodes, nn, sn);
    if (si < 0) die("transfer: unknown source node %s", sn);
    uj_node *st = node_op(&nodes[si], "stat", sp);
    if (!st || !uj_bool(st, "exists", 0) || uj_bool(st, "dir", 0)) die("source %s:%s: not a readable file", sn, sp);
    const uint64_t size = (uint64_t)uj_num(st, "size", 0);
    const int64_t mtime = (int64_t)uj_num(st, "mtime", 0);
    uj_free(st);
    const uj_node *shard = uj_get(t, "shard");
    if (!shard) {
      char dn[64], dp[PATH_MAX];
      if (split_spec(uj_str(t, "dst", ""), dn, dp)) die("transfer: \"dst\" must be node:/absolute/path");
      item_t *it = &items[ni++];
      snprintf(it->src_node, 64, "%s", sn);
      snprintf(it->dst_node, 64, "%s", dn);
      snprintf(it->src_path, PATH_MAX, "%s", sp);
      snprintf(it->dst_path, PATH_MAX, "%s", dp);
      it->offset = (uint64_t)uj_num(t, "offset", 0);
      it->length = uj_get(t, "length") ? (uint64_t)uj_num(t, "length", 0) : size - it->offset;
      if (it->offset + it->length > size) die("transfer %s: range past the end of the source", sp);
      it->src_size = size;
      it->src_mtime = mtime;
      continue;
    }
    /* shard: contiguous ranges by weight, 1 MiB aligned, one file per destination, plus a manifest */
    const uj_node *to = uj_get(shard, "to"), *w = uj_get(shard, "weights");
    const int k = uj_len(to);
    if (!k || (w && uj_len(w) != k)) die("shard: \"to\" needs nodes and \"weights\" one weight per node");
    double total = 0, weight[MAX_NODES];
    int idx = 0;
    for (const uj_node *x = w ? w->child : NULL; x; x = x->next) weight[idx++] = x->num;
    for (int i = 0; i < k; i++) total += w ? weight[i] : 1;
    const char *dst_fmt = uj_str(shard, "dst", NULL);
    if (!dst_fmt || !strstr(dst_fmt, "{node}") || !strstr(dst_fmt, "{index}")) die("shard: \"dst\" must contain {node} and {index}");
    uint64_t at = 0;
    double acc = 0;
    idx = 0;
    for (const uj_node *x = to->child; x; x = x->next, idx++) {
      acc += w ? weight[idx] : 1;
      uint64_t end = idx == k - 1 ? size : ((uint64_t)(size * (acc / total)) >> 20) << 20;
      if (end < at) end = at;
      item_t *it = &items[ni++];
      snprintf(it->src_node, 64, "%s", sn);
      snprintf(it->dst_node, 64, "%s", x->str);
      snprintf(it->src_path, PATH_MAX, "%s", sp);
      /* expand {node} and {index} */
      char out[PATH_MAX] = "", *o = out;
      for (const char *p = dst_fmt; *p && o < out + PATH_MAX - 32;) {
        if (!strncmp(p, "{node}", 6)) { o += snprintf(o, (size_t)(out + PATH_MAX - o), "%s", x->str); p += 6; }
        else if (!strncmp(p, "{index}", 7)) { o += snprintf(o, (size_t)(out + PATH_MAX - o), "%d", idx); p += 7; }
        else *o++ = *p++;
      }
      *o = 0;
      char dn[64], dp[PATH_MAX];
      if (split_spec(out, dn, dp)) die("shard: expanded \"dst\" %s is not node:/path", out);
      snprintf(it->dst_path, PATH_MAX, "%s", dp);
      if (strcmp(dn, x->str)) die("shard: \"dst\" node prefix must be {node}");
      it->offset = at;
      it->length = end - at;
      it->src_size = size;
      it->src_mtime = mtime;
      at = end;
    }
    if (uj_str(shard, "manifest", NULL)) {
      uj_put(&manifest, "%s{\"manifest\":", nmanifest++ ? "," : "");
      uj_put_str(&manifest, uj_str(shard, "manifest", ""));
      uj_put(&manifest, ",\"source\":");
      uj_put_str(&manifest, sp);
      uj_put(&manifest, ",\"size\":%llu,\"mtime\":%lld,\"shards\":[", (unsigned long long)size, (long long)mtime);
      for (int i = ni - k; i < ni; i++) {
        uj_put(&manifest, "%s{\"index\":%d,\"node\":", i > ni - k ? "," : "", i - (ni - k));
        uj_put_str(&manifest, items[i].dst_node);
        uj_put(&manifest, ",\"path\":");
        uj_put_str(&manifest, items[i].dst_path);
        uj_put(&manifest, ",\"offset\":%llu,\"length\":%llu}", (unsigned long long)items[i].offset, (unsigned long long)items[i].length);
      }
      uj_put(&manifest, "]}");
    }
  }
  if (!ni) die("no transfers");

  /* preflight: destinations must not exist (unless overwrite), and each filesystem must hold its bytes plus a reserve */
  const int overwrite = uj_bool(cfg, "overwrite", 0);
  const uint64_t reserve = (uint64_t)(uj_num(cfg, "reserve_gib", 4) * (1ull << 30));
  struct { char node[64]; uint64_t dev, need, free; } fsneed[64];
  int nfs = 0, problems = 0;
  for (int i = 0; i < ni; i++) {
    const int di = find_node(nodes, nn, items[i].dst_node);
    if (di < 0) die("transfer: unknown destination node %s", items[i].dst_node);
    uj_node *st = node_op(&nodes[di], "stat", items[i].dst_path);
    if (!st || uj_get(st, "error")) {
      fprintf(stderr, "ucfile: %s:%s: %s\n", items[i].dst_node, items[i].dst_path, st ? uj_str(st, "error", "") : "no answer");
      problems++;
      uj_free(st);
      continue;
    }
    if (uj_bool(st, "exists", 0) && !overwrite) {
      fprintf(stderr, "ucfile: %s:%s exists (set \"overwrite\": true)\n", items[i].dst_node, items[i].dst_path);
      problems++;
    }
    const uint64_t dev = (uint64_t)uj_num(st, "dev", 0), part = (uint64_t)uj_num(st, "part", 0);
    int k = 0;
    while (k < nfs && !(fsneed[k].dev == dev && !strcmp(fsneed[k].node, items[i].dst_node))) k++;
    if (k == nfs && nfs < 64) {
      snprintf(fsneed[nfs].node, 64, "%s", items[i].dst_node);
      fsneed[nfs].dev = dev;
      fsneed[nfs].need = 0;
      fsneed[nfs].free = (uint64_t)uj_num(st, "free", 0);
      nfs++;
    }
    /* a resumable .part already holds its allocation */
    if (k < 64) fsneed[k].need += items[i].length > part ? items[i].length - part : 0;
    uj_free(st);
  }
  for (int k = 0; k < nfs; k++)
    if (fsneed[k].need + reserve > fsneed[k].free) {
      fprintf(stderr, "ucfile: %s: needs %.2f GB plus %.1f GiB reserve, has %.2f GB free\n", fsneed[k].node,
              fsneed[k].need / 1e9, reserve / 1073741824.0, fsneed[k].free / 1e9);
      problems++;
    }

  /* ranks: self is 0 (the ucomm bootstrap root), then the other nodes that take part */
  node_t *ranked[MAX_NODES];
  int world = 0;
  ranked[world++] = &nodes[find_node(nodes, nn, self)];
  for (int i = 0; i < nn; i++) {
    if (nodes[i].self) continue;
    int used = 0;
    for (int k = 0; k < ni; k++) used |= !strcmp(items[k].src_node, nodes[i].name) || !strcmp(items[k].dst_node, nodes[i].name);
    if (used) ranked[world++] = &nodes[i];
  }
  /* bootstrap address (rank 0 listens there): explicit "root_addr", else the local address the kernel routes each
     agent through; when they differ (e.g. one node only on Ethernet), the non-IPoIB one, which every node can reach */
  char root[128], routed[INET_ADDRSTRLEN] = "";
  const int data_port = (int)uj_num(cfg, "data_port", g_port + 1);
  const char *explicit_root = uj_str(cfg, "root_addr", NULL);
  if (explicit_root) snprintf(routed, sizeof routed, "%s", explicit_root);
  else {
    ifaddr4 la[32];
    const int nla = local_addrs(la, 32);
    int differ = 0;
    for (int r = 1; r < world; r++) {
      char here[INET_ADDRSTRLEN] = "";
      if (local_route(ranked[r]->host, here)) continue;
      int ib = 0;
      for (int i = 0; i < nla; i++)
        if (!strcmp(la[i].addr, here)) ib = is_ib_ifname(la[i].ifname);
      if (!routed[0]) snprintf(routed, sizeof routed, "%s", here);
      else if (strcmp(routed, here)) { differ = 1; if (!ib) snprintf(routed, sizeof routed, "%s", here); }
    }
    if (differ) fprintf(stderr, "ucfile: the nodes are reached through different local addresses; bootstrapping on %s (set \"root_addr\" to choose)\n", routed);
    if (!routed[0]) snprintf(routed, sizeof routed, "127.0.0.1");
  }
  snprintf(root, sizeof root, "%s:%d", routed, data_port);

  /* the expanded job every rank runs */
  uj_buf job = {0};
  uj_put(&job, "{\"world\":%d,\"root\":\"%s\",\"backend\":", world, root);
  uj_put_str(&job, uj_str(cfg, "backend", "auto"));
  uj_put(&job, ",\"chunk_bytes\":%llu,\"depth\":%d,\"verify\":", (unsigned long long)(uj_num(cfg, "chunk_mib", 16) * (1 << 20)),
         (int)uj_num(cfg, "depth", 4));
  uj_put_str(&job, uj_str(cfg, "verify", "none"));
  uj_put(&job, ",\"overwrite\":%s,\"keep_mtime\":%s,\"timeout_ms\":%d,\"ranks\":[", overwrite ? "true" : "false",
         uj_bool(cfg, "keep_mtime", 1) ? "true" : "false", (int)uj_num(cfg, "timeout_ms", 120000));
  for (int r = 0; r < world; r++) { uj_put(&job, "%s", r ? "," : ""); uj_put_str(&job, ranked[r]->name); }
  uj_put(&job, "],\"items\":[");
  uint64_t total = 0;
  for (int i = 0; i < ni; i++) {
    uj_put(&job, "%s{\"src_node\":", i ? "," : "");
    uj_put_str(&job, items[i].src_node);
    uj_put(&job, ",\"dst_node\":");
    uj_put_str(&job, items[i].dst_node);
    uj_put(&job, ",\"src_path\":");
    uj_put_str(&job, items[i].src_path);
    uj_put(&job, ",\"dst_path\":");
    uj_put_str(&job, items[i].dst_path);
    uj_put(&job, ",\"offset\":%llu,\"length\":%llu,\"src_size\":%llu,\"src_mtime\":%lld}", (unsigned long long)items[i].offset,
           (unsigned long long)items[i].length, (unsigned long long)items[i].src_size, (long long)items[i].src_mtime);
    total += items[i].length;
  }
  uj_put(&job, "]}");
  if (dry_run || problems) {
    printf("{\"plan\":%s,\"total_bytes\":%llu,\"problems\":%d%s%s%s}\n", job.s, (unsigned long long)total, problems,
           nmanifest ? ",\"manifests\":[" : "", nmanifest ? manifest.s : "", nmanifest ? "]" : "");
    return problems ? 2 : 0;
  }

  /* start the agents (each answers "accepted", then runs its rank), run rank 0 here, collect results */
  int fds[MAX_NODES];
  for (int r = 1; r < world; r++) {
    uj_buf req = {0};
    uj_put(&req, "{\"op\":\"job\",\"rank\":%d,\"job\":%s}", r, job.s);
    fds[r] = tcp_connect(ranked[r]->host, ranked[r]->port, 5000);
    if (fds[r] < 0 || send_line(fds[r], &req)) die("cannot reach the agent on %s (%s:%d)", ranked[r]->name, ranked[r]->host, ranked[r]->port);
    free(req.s);
    char *ack = recv_line(fds[r], 15000);
    if (!ack || !strstr(ack, "\"accepted\"")) die("agent %s refused the job: %s", ranked[r]->name, ack ? ack : "no answer");
    free(ack);
  }
  job_t j;
  char perr[128];
  uj_node *jv = uj_parse(job.s, job.n, NULL);
  if (!jv || parse_job(jv, &j, perr, sizeof perr)) die("internal: %s", perr);
  const double t0 = now_s();
  uj_buf mine = {0};
  int failures = run_job(&j, 0, -1, &mine);
  printf("{\"total_bytes\":%llu,\"nodes\":[{\"node\":\"%s\",\"failures\":%d,\"results\":%s}", (unsigned long long)total, self,
         failures, mine.s);
  for (int r = 1; r < world; r++) {
    char *line;
    uj_node *v = NULL;
    while ((line = recv_line(fds[r], j.timeout_ms > 0 ? j.timeout_ms * 4 : 0))) {
      v = uj_parse(line, strlen(line), NULL);
      if (v && uj_get(v, "progress")) {
        if (g_verbose) fprintf(stderr, "%s: %s\n", ranked[r]->name, line);
        uj_free(v);
        v = NULL;
        free(line);
        continue;
      }
      printf(",%s", line);
      free(line);
      break;
    }
    if (!v) { printf(",{\"node\":\"%s\",\"error\":\"no result\"}", ranked[r]->name); failures++; }
    else failures += (int)uj_num(v, "failures", 0);
    uj_free(v);
    close(fds[r]);
  }
  const double secs = now_s() - t0;
  printf("],\"elapsed\":%.3f,\"GBps\":%.3f,\"failures\":%d}\n", secs, secs > 0 ? total / secs / 1e9 : 0.0, failures);
  /* manifests (written by the coordinator, next to it) once every shard arrived */
  if (nmanifest && !failures) {
    uj_node *mv = NULL;
    uj_buf all = {0};
    uj_put(&all, "[%s]", manifest.s);
    mv = uj_parse(all.s, all.n, NULL);
    for (const uj_node *m = mv ? mv->child : NULL; m; m = m->next) {
      FILE *out = fopen(uj_str(m, "manifest", ""), "w");
      if (!out) { fprintf(stderr, "ucfile: cannot write %s\n", uj_str(m, "manifest", "")); continue; }
      uj_buf one = {0};
      uj_dump(&one, m);
      fprintf(out, "%s\n", one.s);
      fclose(out);
      free(one.s);
    }
    uj_free(mv);
    free(all.s);
  }
  return failures ? 1 : 0;
}

/* ------------------------------------------------------------------ main */

static void usage(void) {
  fprintf(stderr,
          "usage: ucfile agent --root DIR [--root DIR...] [--name N] [--port P] [-v]\n"
          "       ucfile discover [--port P] [--timeout MS] [--to ADDR...]\n"
          "       ucfile run JOB.json [--dry-run] [-v]\n"
          "       ucfile df|ls|stat|rm|mkdir NODE [PATH...] [--port P]\n");
  exit(2);
}

int main(int argc, char **argv) {
  if (argc < 2) usage();
  gethostname(g_name, sizeof g_name);
  char *dot = strchr(g_name, '.');
  if (dot) *dot = 0;
  const char *cmd = argv[1];
  char *args[64];
  const char *extra[16];
  int nargs = 0, nextra = 0, dry = 0, timeout = 1000;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--root") && i + 1 < argc && g_nroots < MAX_ROOTS) g_roots[g_nroots++] = argv[++i];
    else if (!strcmp(argv[i], "--name") && i + 1 < argc) snprintf(g_name, sizeof g_name, "%s", argv[++i]);
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) g_port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--to") && i + 1 < argc && nextra < 16) extra[nextra++] = argv[++i];
    else if (!strcmp(argv[i], "--dry-run")) dry = 1;
    else if (!strcmp(argv[i], "-v")) g_verbose = 1;
    else if (nargs < 64) args[nargs++] = argv[i];
  }
  if (!strcmp(cmd, "agent")) return cmd_agent();
  if (!strcmp(cmd, "discover")) {
    peer_t peers[32];
    int n = discover(g_port, timeout, extra, nextra, peers, 32);
    printf("[");
    for (int i = 0; i < n; i++) {
      uj_buf b = {0};
      uj_dump(&b, peers[i].info);
      printf("%s{\"reply_from\":\"%s\",\"via_ipoib\":%s,\"info\":%s}", i ? "," : "", peers[i].addr,
             peers[i].ipoib ? "true" : "false", b.s);
      free(b.s);
      uj_free(peers[i].info);
    }
    printf("]\n");
    return n ? 0 : 1;
  }
  if (!strcmp(cmd, "run")) {
    if (nargs != 1) usage();
    return cmd_run(args[0], dry);
  }
  if (!strcmp(cmd, "df") || !strcmp(cmd, "ls") || !strcmp(cmd, "stat") || !strcmp(cmd, "rm") || !strcmp(cmd, "mkdir")) {
    if (nargs < 1) usage();
    char host[INET_ADDRSTRLEN + 64];
    int port = g_port;
    if (resolve_node(args[0], host, &port)) die("node %s: no agent answered discovery", args[0]);
    if (!strcmp(cmd, "df")) {
      uj_buf b = {0};
      uj_put(&b, "{\"op\":\"info\"}");
      uj_node *r = request(host, port, &b, 15000);
      if (!r) die("no answer from %s", args[0]);
      uj_buf o = {0};
      uj_dump(&o, uj_get(r, "roots"));
      printf("%s\n", o.s);
      return 0;
    }
    int bad = 0;
    for (int i = 1; i < nargs; i++) {
      uj_buf b = {0};
      uj_put(&b, "{\"op\":\"%s\",\"path\":", cmd);
      uj_put_str(&b, args[i]);
      uj_put(&b, "}");
      uj_node *r = request(host, port, &b, 15000);
      uj_buf o = {0};
      uj_dump(&o, r);
      printf("%s\n", o.s);
      bad += !r || uj_get(r, "error") != NULL;
      uj_free(r);
      free(b.s);
      free(o.s);
    }
    return bad ? 1 : 0;
  }
  usage();
  return 2;
}
