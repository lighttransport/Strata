**Findings (concrete bugs, not style):**

1. **`out` mutated on failure** — `out = 0` runs before validation, and digits accumulate before a later bad char. Trigger: `out = 7; parse_u64("12a", out)` → returns false but `out == 12`, violating "out unchanged on every failure."
2. **Empty input accepted** — `parse_u64("", out)` returns true; contract requires nonempty ASCII digits. Trigger: `s == ""`.
3. **Overflow wraps silently** — `out*10 + d` has no UINT64_MAX guard, so values above the max wrap and the function returns true. Trigger: `"1844674407378"` (max+1) or `"99999999999999999999"` → true, garbage value.
4. **Dangling retained views** — `get` returns a `string_view` into `names`; a later