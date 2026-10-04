#define _GNU_SOURCE
#include "ibvew.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

int ibvew_load(ibvew_api *api) {
  memset(api, 0, sizeof *api);
  const char *names[] = {getenv("UCOMM_IBVERBS_LIB"), "libibverbs.so.1", "libibverbs.so"};
  for (size_t i = 0; i < sizeof names / sizeof names[0] && !api->lib; i++)
    if (names[i] && *names[i]) api->lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
  if (!api->lib) return -1;

#define LOAD(field, sym)                                       \
  do {                                                         \
    void *p_ = dlsym(api->lib, sym);                           \
    if (!p_) goto fail;                                        \
    memcpy(&api->field, &p_, sizeof p_); /* object->fn ptr */  \
  } while (0)
  LOAD(get_device_list, "ibv_get_device_list");
  LOAD(free_device_list, "ibv_free_device_list");
  LOAD(get_device_name, "ibv_get_device_name");
  LOAD(open_device, "ibv_open_device");
  LOAD(close_device, "ibv_close_device");
  LOAD(query_port, "ibv_query_port");
  LOAD(query_gid, "ibv_query_gid");
  LOAD(alloc_pd, "ibv_alloc_pd");
  LOAD(dealloc_pd, "ibv_dealloc_pd");
  LOAD(reg_mr, "ibv_reg_mr");
  LOAD(dereg_mr, "ibv_dereg_mr");
  LOAD(create_cq, "ibv_create_cq");
  LOAD(destroy_cq, "ibv_destroy_cq");
  LOAD(create_qp, "ibv_create_qp");
  LOAD(modify_qp, "ibv_modify_qp");
  LOAD(destroy_qp, "ibv_destroy_qp");
#undef LOAD
  return 0;
fail:
  ibvew_unload(api);
  return -1;
}

void ibvew_unload(ibvew_api *api) {
  if (api->lib) dlclose(api->lib);
  memset(api, 0, sizeof *api);
}
