Trial division with 6k±1.</think>```cpp
#include <cstdint>

bool is_prime(unsigned n) {
    if (n < 2) return false;
    if (n < 4) return true;          // 2, 3
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}
```