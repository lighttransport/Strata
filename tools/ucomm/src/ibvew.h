/* ibvew: ibverbs extension wrangler.
 *
 * The subset of the libibverbs ABI that ucomm needs, written out by hand so no verbs headers are required,
 * plus a loader that resolves the entry points from libibverbs.so.1 with dlopen/dlsym (the cuew approach).
 * Layouts follow rdma-core's <infiniband/verbs.h>, which kept the libibverbs 1.1 ABI that ConnectX-3 (mlx4)
 * era stacks ship. Only leading fields that ucomm reads are named; trailing fields are left as padding.
 * post_send / post_recv / poll_cq are inline functions in verbs.h that call through ibv_context.ops, so the
 * same is done here. */
#ifndef IBVEW_H_
#define IBVEW_H_

#include <stddef.h>
#include <stdint.h>

struct ibv_device;
struct ibv_srq;
struct ibv_ah;
struct ibv_mw;
struct ibv_comp_channel;
struct ibv_cq;
struct ibv_qp;
struct ibv_wc;
struct ibv_send_wr;
struct ibv_recv_wr;

union ibv_gid {
  uint8_t raw[16];
  struct {
    uint64_t subnet_prefix;
    uint64_t interface_id;
  } global;
};

enum { IBV_PORT_ACTIVE = 4 };
enum { IBV_LINK_LAYER_UNSPECIFIED = 0, IBV_LINK_LAYER_INFINIBAND = 1, IBV_LINK_LAYER_ETHERNET = 2 };
enum { IBV_MTU_256 = 1, IBV_MTU_512, IBV_MTU_1024, IBV_MTU_2048, IBV_MTU_4096 };
enum { IBV_ACCESS_LOCAL_WRITE = 1, IBV_ACCESS_REMOTE_WRITE = 2, IBV_ACCESS_REMOTE_READ = 4 };
enum { IBV_QPT_RC = 2 };
enum { IBV_QPS_RESET = 0, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS };
enum {
  IBV_QP_STATE = 1 << 0,
  IBV_QP_ACCESS_FLAGS = 1 << 3,
  IBV_QP_PKEY_INDEX = 1 << 4,
  IBV_QP_PORT = 1 << 5,
  IBV_QP_AV = 1 << 7,
  IBV_QP_PATH_MTU = 1 << 8,
  IBV_QP_TIMEOUT = 1 << 9,
  IBV_QP_RETRY_CNT = 1 << 10,
  IBV_QP_RNR_RETRY = 1 << 11,
  IBV_QP_RQ_PSN = 1 << 12,
  IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13,
  IBV_QP_MIN_RNR_TIMER = 1 << 15,
  IBV_QP_SQ_PSN = 1 << 16,
  IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
  IBV_QP_DEST_QPN = 1 << 20
};
enum { IBV_WR_RDMA_WRITE = 0, IBV_WR_RDMA_WRITE_WITH_IMM, IBV_WR_SEND, IBV_WR_SEND_WITH_IMM, IBV_WR_RDMA_READ };
enum { IBV_SEND_FENCE = 1, IBV_SEND_SIGNALED = 2, IBV_SEND_SOLICITED = 4, IBV_SEND_INLINE = 8 };
enum { IBV_WC_SUCCESS = 0 };

struct ibv_context_ops { /* 33 slots; libibverbs 1.1 and rdma-core agree on positions */
  void *slot0_10[11];
  int (*poll_cq)(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc); /* 11 */
  void *slot12_24[13];
  int (*post_send)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad); /* 25 */
  int (*post_recv)(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad); /* 26 */
  void *slot27_32[6];
};

struct ibv_context {
  struct ibv_device *device;
  struct ibv_context_ops ops;
  /* cmd_fd, async_fd, num_comp_vectors, mutex, abi_compat follow */
};

struct ibv_pd {
  struct ibv_context *context;
  uint32_t handle;
};

struct ibv_mr {
  struct ibv_context *context;
  struct ibv_pd *pd;
  void *addr;
  size_t length;
  uint32_t handle;
  uint32_t lkey;
  uint32_t rkey;
};

struct ibv_cq {
  struct ibv_context *context;
  /* channel, cq_context, handle, cqe, mutex, cond, ... */
};

struct ibv_qp {
  struct ibv_context *context;
  void *qp_context;
  struct ibv_pd *pd;
  struct ibv_cq *send_cq;
  struct ibv_cq *recv_cq;
  struct ibv_srq *srq;
  uint32_t handle;
  uint32_t qp_num;
  /* state, qp_type, mutex, cond, events_completed */
};

struct ibv_port_attr {
  int state;
  int max_mtu;
  int active_mtu;
  int gid_tbl_len;
  uint32_t port_cap_flags;
  uint32_t max_msg_sz;
  uint32_t bad_pkey_cntr;
  uint32_t qkey_viol_cntr;
  uint16_t pkey_tbl_len;
  uint16_t lid;
  uint16_t sm_lid;
  uint8_t lmc;
  uint8_t max_vl_num;
  uint8_t sm_sl;
  uint8_t subnet_timeout;
  uint8_t init_type_reply;
  uint8_t active_width;
  uint8_t active_speed;
  uint8_t phys_state;
  uint8_t link_layer;
  uint8_t flags;
  uint16_t port_cap_flags2;
  uint8_t reserved[64]; /* room for fields added by newer rdma-core */
};

struct ibv_sge {
  uint64_t addr;
  uint32_t length;
  uint32_t lkey;
};

struct ibv_mw_bind_info {
  struct ibv_mr *mr;
  uint64_t addr;
  uint64_t length;
  unsigned int mw_access_flags;
};

struct ibv_send_wr {
  uint64_t wr_id;
  struct ibv_send_wr *next;
  struct ibv_sge *sg_list;
  int num_sge;
  int opcode;
  unsigned int send_flags;
  uint32_t imm_data;
  union {
    struct {
      uint64_t remote_addr;
      uint32_t rkey;
    } rdma;
    struct {
      uint64_t remote_addr;
      uint64_t compare_add;
      uint64_t swap;
      uint32_t rkey;
    } atomic;
    struct {
      struct ibv_ah *ah;
      uint32_t remote_qpn;
      uint32_t remote_qkey;
    } ud;
  } wr;
  union {
    struct {
      uint32_t remote_srqn;
    } xrc;
  } qp_type;
  union {
    struct {
      struct ibv_mw *mw;
      uint32_t rkey;
      struct ibv_mw_bind_info bind_info;
    } bind_mw;
    struct {
      void *hdr;
      uint16_t hdr_sz;
      uint16_t mss;
    } tso;
  } ext;
};

struct ibv_recv_wr {
  uint64_t wr_id;
  struct ibv_recv_wr *next;
  struct ibv_sge *sg_list;
  int num_sge;
};

struct ibv_wc {
  uint64_t wr_id;
  int status;
  int opcode;
  uint32_t vendor_err;
  uint32_t byte_len;
  uint32_t imm_data;
  uint32_t qp_num;
  uint32_t src_qp;
  unsigned int wc_flags;
  uint16_t pkey_index;
  uint16_t slid;
  uint8_t sl;
  uint8_t dlid_path_bits;
};

struct ibv_qp_cap {
  uint32_t max_send_wr;
  uint32_t max_recv_wr;
  uint32_t max_send_sge;
  uint32_t max_recv_sge;
  uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
  void *qp_context;
  struct ibv_cq *send_cq;
  struct ibv_cq *recv_cq;
  struct ibv_srq *srq;
  struct ibv_qp_cap cap;
  int qp_type;
  int sq_sig_all;
};

struct ibv_global_route {
  union ibv_gid dgid;
  uint32_t flow_label;
  uint8_t sgid_index;
  uint8_t hop_limit;
  uint8_t traffic_class;
};

struct ibv_ah_attr {
  struct ibv_global_route grh;
  uint16_t dlid;
  uint8_t sl;
  uint8_t src_path_bits;
  uint8_t static_rate;
  uint8_t is_global;
  uint8_t port_num;
};

struct ibv_qp_attr {
  int qp_state;
  int cur_qp_state;
  int path_mtu;
  int path_mig_state;
  uint32_t qkey;
  uint32_t rq_psn;
  uint32_t sq_psn;
  uint32_t dest_qp_num;
  unsigned int qp_access_flags;
  struct ibv_qp_cap cap;
  struct ibv_ah_attr ah_attr;
  struct ibv_ah_attr alt_ah_attr;
  uint16_t pkey_index;
  uint16_t alt_pkey_index;
  uint8_t en_sqd_async_notify;
  uint8_t sq_draining;
  uint8_t max_rd_atomic;
  uint8_t max_dest_rd_atomic;
  uint8_t min_rnr_timer;
  uint8_t port_num;
  uint8_t timeout;
  uint8_t retry_cnt;
  uint8_t rnr_retry;
  uint8_t alt_port_num;
  uint8_t alt_timeout;
  uint32_t rate_limit;
};

typedef struct {
  void *lib;
  struct ibv_device **(*get_device_list)(int *num);
  void (*free_device_list)(struct ibv_device **list);
  const char *(*get_device_name)(struct ibv_device *dev);
  struct ibv_context *(*open_device)(struct ibv_device *dev);
  int (*close_device)(struct ibv_context *ctx);
  int (*query_port)(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr);
  int (*query_gid)(struct ibv_context *ctx, uint8_t port, int index, union ibv_gid *gid);
  struct ibv_pd *(*alloc_pd)(struct ibv_context *ctx);
  int (*dealloc_pd)(struct ibv_pd *pd);
  struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t len, int access);
  int (*dereg_mr)(struct ibv_mr *mr);
  struct ibv_cq *(*create_cq)(struct ibv_context *ctx, int cqe, void *cq_ctx, struct ibv_comp_channel *ch,
                              int comp_vector);
  int (*destroy_cq)(struct ibv_cq *cq);
  struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *attr);
  int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask);
  int (*destroy_qp)(struct ibv_qp *qp);
} ibvew_api;

/* Loads libibverbs (env UCOMM_IBVERBS_LIB overrides the soname). 0 on success. */
int ibvew_load(ibvew_api *api);
void ibvew_unload(ibvew_api *api);

static inline int ibvew_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc) {
  return cq->context->ops.poll_cq(cq, n, wc);
}
static inline int ibvew_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
  return qp->context->ops.post_send(qp, wr, bad);
}
static inline int ibvew_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad) {
  return qp->context->ops.post_recv(qp, wr, bad);
}

#endif
