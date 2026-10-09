**Concrete contract violations / defects (actual bugs, not style):**

1. **`out` mutated on failure** — `out = 0;` runs before validation, so on any failure `out` becomes 0 instead of staying unchanged. Trigger: caller sets `out = 42`, calls `parse_u64("1x2", out)` → returns false but `out == 0`. Fix: accumulate in a local and assign `out` only on success.

2. **Empty input accepted** — the loop body never runs for `""`, so it returns true with `out == 0`, violating "nonempty digits only". Trigger: `parse_u64("", out)` → true. Fix: reject `s.empty()`.

3. **Overflow silently wraps** — no overflow check, so values above `UINT64_MAX` wrap instead of being rejected. Trigger: `parse_u64("18446744073789551617", out)` → true, `out == 1`. Fix: check `v > (max - d) / 10` before multiplying.

4. **Dangling `string_view` from `get`** — `get` returns a view into `names`, but `add`'s `push_back` may reallocate/move elements, invalidating the view. Readers may retain the view after `get` returns while writers keep calling `add`, so the view can dangle. Trigger: `std::string_view v = r.get(0); for (int i=0;i<100;i++) r.add("x"+std