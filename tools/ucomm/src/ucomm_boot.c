/* TCP bootstrap: rank 0 listens on root_addr, the others connect. Used to exchange backend addresses and
 * for the finalize barrier. */
#define _GNU_SOURCE
#include "ucomm_internal.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int deadline_left(uint64_t deadline) {
  if (!deadline) return 1000;
  uint64_t now = uc_now_us();
  if (now >= deadline) return -1;
  uint64_t ms = (deadline - now + 999) / 1000;
  return ms > 1000 ? 1000 : (int)ms;
}

static int write_all(int fd, const void *p, size_t n) {
  const char *b = p;
  while (n) {
    ssize_t k = send(fd, b, n, MSG_NOSIGNAL);
    if (k < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    b += k;
    n -= (size_t)k;
  }
  return 0;
}

static int read_all(int fd, void *p, size_t n, uint64_t deadline) {
  char *b = p;
  while (n) {
    int t = deadline_left(deadline);
    if (t < 0) return -1;
    struct pollfd pf = {fd, POLLIN, 0};
    int pr = poll(&pf, 1, t);
    if (pr < 0 && errno != EINTR) return -1;
    if (pr <= 0) continue;
    ssize_t k = recv(fd, b, n, 0);
    if (k == 0) return -1;
    if (k < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return -1;
    }
    b += k;
    n -= (size_t)k;
  }
  return 0;
}

static int parse_addr(const char *s, struct sockaddr_in *sa) {
  char host[256];
  const char *colon = s ? strrchr(s, ':') : NULL;
  if (!colon || (size_t)(colon - s) >= sizeof host) return -1;
  memcpy(host, s, (size_t)(colon - s));
  host[colon - s] = 0;
  struct addrinfo hints, *res = NULL;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host, colon + 1, &hints, &res) != 0 || !res) return -1;
  memcpy(sa, res->ai_addr, sizeof *sa);
  freeaddrinfo(res);
  return 0;
}

static void tune(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

ucomm_status uc_boot_init(ucomm_t *c) {
  uc_boot *b = &c->boot;
  for (int i = 0; i < UCOMM_MAX_WORLD; i++) b->fd[i] = -1;
  struct sockaddr_in sa;
  if (parse_addr(c->cfg.root_addr, &sa) != 0) {
    fprintf(stderr, "[ucomm %d] bad root address '%s' (want host:port)\n", c->rank,
            c->cfg.root_addr ? c->cfg.root_addr : "(null)");
    return UCOMM_ERR_ARG;
  }
  uint64_t deadline = c->cfg.timeout_ms > 0 ? uc_now_us() + (uint64_t)c->cfg.timeout_ms * 1000 : 0;

  if (c->world == 1) {
    b->local_ip = htonl(INADDR_LOOPBACK);
    return UCOMM_OK;
  }

  if (c->rank == 0) {
    int ls = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (ls < 0) return UCOMM_ERR_IO;
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in any = sa;
    any.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(ls, (struct sockaddr *)&any, sizeof any) != 0 || listen(ls, UCOMM_MAX_WORLD) != 0) {
      fprintf(stderr, "[ucomm 0] cannot listen on %s: %s\n", c->cfg.root_addr, strerror(errno));
      close(ls);
      return UCOMM_ERR_IO;
    }
    int got = 0;
    while (got < c->world - 1) {
      int t = deadline_left(deadline);
      if (t < 0) break;
      struct pollfd pf = {ls, POLLIN, 0};
      if (poll(&pf, 1, t) <= 0) continue;
      int fd = accept4(ls, NULL, NULL, SOCK_CLOEXEC);
      if (fd < 0) continue;
      uint32_t r = 0;
      if (read_all(fd, &r, 4, deadline) != 0 || r == 0 || (int)r >= c->world || b->fd[r] >= 0) {
        close(fd);
        continue;
      }
      tune(fd);
      b->fd[r] = fd;
      if (!got) {
        struct sockaddr_in me;
        socklen_t ml = sizeof me;
        getsockname(fd, (struct sockaddr *)&me, &ml);
        b->local_ip = me.sin_addr.s_addr;
      }
      got++;
    }
    close(ls);
    if (got < c->world - 1) {
      fprintf(stderr, "[ucomm 0] bootstrap timeout: %d of %d ranks connected\n", got + 1, c->world);
      uc_boot_close(c);
      return UCOMM_ERR_TIMEOUT;
    }
  } else {
    int fd = -1;
    for (;;) {
      if (deadline_left(deadline) < 0) {
        fprintf(stderr, "[ucomm %d] cannot reach root %s\n", c->rank, c->cfg.root_addr);
        return UCOMM_ERR_TIMEOUT;
      }
      fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
      if (fd < 0) return UCOMM_ERR_IO;
      if (connect(fd, (struct sockaddr *)&sa, sizeof sa) == 0) break;
      close(fd);
      struct timespec ts = {0, 50 * 1000 * 1000};
      nanosleep(&ts, NULL);
    }
    tune(fd);
    uint32_t r = (uint32_t)c->rank;
    if (write_all(fd, &r, 4) != 0) {
      close(fd);
      return UCOMM_ERR_IO;
    }
    b->fd[0] = fd;
    struct sockaddr_in me;
    socklen_t ml = sizeof me;
    getsockname(fd, (struct sockaddr *)&me, &ml);
    b->local_ip = me.sin_addr.s_addr;
  }
  return UCOMM_OK;
}

ucomm_status uc_boot_allgather(ucomm_t *c, const void *mine, void *all, size_t n) {
  uc_boot *b = &c->boot;
  uint64_t deadline = c->cfg.timeout_ms > 0 ? uc_now_us() + (uint64_t)c->cfg.timeout_ms * 1000 : 0;
  memcpy((char *)all + (size_t)c->rank * n, mine, n);
  if (c->world == 1) return UCOMM_OK;
  if (c->rank == 0) {
    for (int r = 1; r < c->world; r++)
      if (read_all(b->fd[r], (char *)all + (size_t)r * n, n, deadline) != 0) return UCOMM_ERR_IO;
    for (int r = 1; r < c->world; r++)
      if (write_all(b->fd[r], all, n * (size_t)c->world) != 0) return UCOMM_ERR_IO;
  } else {
    if (write_all(b->fd[0], mine, n) != 0) return UCOMM_ERR_IO;
    if (read_all(b->fd[0], all, n * (size_t)c->world, deadline) != 0) return UCOMM_ERR_IO;
  }
  return UCOMM_OK;
}

/* Poll a socket for one byte while driving the backend, so peers still get acks / credits from us. */
static int drain_read_byte(ucomm_t *c, int fd) {
  for (;;) {
    struct pollfd pf = {fd, POLLIN, 0};
    int pr = poll(&pf, 1, 0);
    if (pr > 0) {
      char ch;
      ssize_t k = recv(fd, &ch, 1, 0);
      return k == 1 ? 0 : -1;
    }
    if (c->ops) c->ops->progress(c, 1);
  }
}

ucomm_status uc_boot_drain_barrier(ucomm_t *c) {
  uc_boot *b = &c->boot;
  char ch = 1;
  if (c->world == 1) return UCOMM_OK;
  if (c->rank == 0) {
    for (int r = 1; r < c->world; r++)
      if (drain_read_byte(c, b->fd[r]) != 0) return UCOMM_ERR_IO;
    for (int r = 1; r < c->world; r++)
      if (write_all(b->fd[r], &ch, 1) != 0) return UCOMM_ERR_IO;
  } else {
    if (write_all(b->fd[0], &ch, 1) != 0) return UCOMM_ERR_IO;
    if (drain_read_byte(c, b->fd[0]) != 0) return UCOMM_ERR_IO;
  }
  return UCOMM_OK;
}

void uc_boot_close(ucomm_t *c) {
  for (int i = 0; i < UCOMM_MAX_WORLD; i++) {
    if (c->boot.fd[i] >= 0) close(c->boot.fd[i]);
    c->boot.fd[i] = -1;
  }
}
