Findings (concrete bugs, not style):

1. **Failure leaves out changed**: `out = 0;` runs before validation, so any failure overwrites the caller's value. Trigger: `std::uint64_t v = 42; parse_u64("12x", v);` → returns false but `v == 0`. Fix: accumulate in a local, assign `out` only on success.

2. **Empty string accepted**: loop over empty `s` succeeds trivially. Trigger: `parse_u64("", v)` → true, `v == 0`. Contract requires nonempty.

3. **Overflow silently wraps**: `out = out*10 + d` wraps mod 2^64, so values above UINT64_MAX are accepted. Trigger: `parse_u64("18446744073709551616", v)` → true, `v == 0`. Fix: reject when `acc > (UINT64_MAX - d)/10`.

4. **Dangling string_view**: `get` returns a view into `names` and releases the lock; a later `add` may `push_back` and reallocate, invalidating retained views. Trigger: reader holds `sv = get(0)`, writer calls `add("...")` enough times to exceed capacity → reader's `sv` dangles. Fix: return `std::string` (copy), or store immutable blocks (e.g. `std::deque` of per-entry chunks / `std::string` per slot) so reallocation never moves retained data.

5. **No bounds check in `get(i)`**: `names[i]` with `i >= names.size()` is UB. Trigger: `get(0)` on an empty registry. Fix: bounds-check and fail/throw.

6. Style (not a bug): `noexcept` on parse is fine once overflow is handled; `std::lock_guard` usage is fine. No invented code needed beyond the fixes above.

```cpp
bool parse_u64(std::string_view s, std::uint64_t& out) noexcept {
    if (s.empty()) return false;
    std::uint64_t acc = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        std::uint64_t d = static_cast<std::uint64_t>(c - '0');
        if (acc > (std::numeric_limits<std::uint64_t>::max() - d) / 10)
            return false;               // would exceed UINT64_MAX
        acc = acc * 10 + d;
    }
    out = acc;                          // assign only on success
    return true;
}
```

For Registry, the safe fix is `std::string get(std::size_t i)` (copy under lock, bounds-checked); if views must be retainable, keep entries in non-relocating storage (deque of blocks) so `add` never invalidates them.