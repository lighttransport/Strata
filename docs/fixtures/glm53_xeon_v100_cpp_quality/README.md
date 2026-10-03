# Xeon Gold / Tesla V100 generated C++ validation

Each 512-token response was compiled as C++17 with `-Wall -Wextra -Werror`
and passed 400,532 independent strict unsigned-decimal parsing cases.
Repeated token IDs were checked separately. Identical sources are stored once.

The [hardware report](../../GLM53_V100.md) describes the measurement method.
The [validation summary](report.json) maps every configuration to its source hash.

| Configuration | Generated source | Oracle cases |
| --- | --- | ---: |
| single_gpu_32768m_17w | [generated-4f20ed5b0ced.cpp](generated-4f20ed5b0ced.cpp) | 400,532 passed |
| single_gpu_32768m_35w | [generated-4f20ed5b0ced.cpp](generated-4f20ed5b0ced.cpp) | 400,532 passed |
| single_gpu_16384m_17w | [generated-1b19910611aa.cpp](generated-1b19910611aa.cpp) | 400,532 passed |
| single_gpu_16384m_35w | [generated-1b19910611aa.cpp](generated-1b19910611aa.cpp) | 400,532 passed |
| single_gpu_15360m_17w | [generated-ad6f499f7183.cpp](generated-ad6f499f7183.cpp) | 400,532 passed |
| single_gpu_15360m_35w | [generated-ad6f499f7183.cpp](generated-ad6f499f7183.cpp) | 400,532 passed |
| dual_socket_1gpu_16384m_35w | [generated-1b19910611aa.cpp](generated-1b19910611aa.cpp) | 400,532 passed |
| dual_socket_2gpu_16384m_35w | [generated-37e10472970d.cpp](generated-37e10472970d.cpp) | 400,532 passed |
| dual_socket_1gpu_16384m_71w | [generated-1b19910611aa.cpp](generated-1b19910611aa.cpp) | 400,532 passed |
| dual_socket_2gpu_16384m_71w | [generated-37e10472970d.cpp](generated-37e10472970d.cpp) | 400,532 passed |
| dual_socket_1gpu_15360m_35w | [generated-ad6f499f7183.cpp](generated-ad6f499f7183.cpp) | 400,532 passed |
| dual_socket_2gpu_15360m_35w | [generated-2ef575b2bc16.cpp](generated-2ef575b2bc16.cpp) | 400,532 passed |
