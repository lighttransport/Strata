**Review of review.cpp (C++17).** Background pool.hpp is context only; the target is the five functions plus the Snapshot control example.

## Concrete correctness bugs

**1. `copy_range` — overflow wrap in `offset + length`.**
Trigger: `input` of 4 chars, `offset = UINT64_MAX - 1`, `length = 2`. `offset + length` wraps to `1`, `1 > 4` is false, so the guard passes and `memcpy` reads unmapped memory past the vector — violating "entirely inside the mapped input". Unsigned addition wraps by definition, so the check is unsound.
Minimal correction:
```cpp
if (length > input.size() || offset > input.size() - length) return false;
```
(equivalently `offset <= size && length <= size - offset`; no wrap possible). Note `input.data() + offset` is then always in range even for a zero-length tail copy.

**2. `display_name` — dangling `std::string_view`.**
Trigger: any call. `result` is a function-local `std::string`; it dies at the closing brace, and the returned view is its dangling referent — the caller cannot use it at all, let alone "until the next invocation". The contract permits a per-call-site storage that survives until the next call, which a local does not.
Minimal correction (fits the contract's lifetime clause exactly):
```cpp
static thread_local std::string result;
result = "worker-" + std::to_string(id);
return result;
```
(Alternative: change the signature to return `std::string`; but the thread-local static is the minimal edit that honours the stated contract.)

**3. `Cache::get` — unsynchronised read of `values_`.**
Trigger: thread A `put(1, "x")` while thread B `get(1)`. `get` does `values_.find` with no lock while `put` mutates the map under `mutex_` — a data race (unordered_map rehash/node rewrite can crash or corrupt the reader). The contract explicitly says get/put run concurrently, so the unsynchronised read is a defect, not an assumption.
Minimal correction: open with `std::lock_guard<std::mutex> lock(mutex_);` before the `find`. The returned `shared_ptr` shares ownership of the string, so it stays usable after later `put`s, as required.

**4. `make_tasks` — tasks capture a dead loop variable.**
Trigger: `auto t = make_tasks(3);` then call `t[1]()` after the loop has ended. Each lambda binds `&i`, but `i` is scoped to one for-iteration and dies when that iteration ends; the reference outlives its referent, so every invocation reads a dangling stack slot (and the captured variable is shared across all closures, so even the "one callback per index" shape is not guaranteed). Fix:
```cpp
tasks.push_back([i = i] { return i; });
```
Capturing by value (moved into the closure) makes each task return its own index, satisfying "exactly one callback per index [0, count) in any order".

**5. `valid_index` — `index == size` reported valid.**
Trigger: `valid_index(0, 0)` returns true, and `valid_index(5, 5)` returns true. `<=` admits the one-p-p position, which is not a valid index; the contract's own note ("empty arrays have no valid index") contradicts the `<=` result for size 0. Fix: `return index < size;`. The "without reading the array" half is already honoured — the function takes only `size`.

## Snapshot::read lifetime assessment (explicit, as requested)

`read` takes the lock, copies the `shared_ptr<const std::string>` (incrementing the refcount while holding `mutex_`), releases, and returns the copy. The returned `shared_ptr` shares ownership of the referent, so the string outlives the call regardless of what `replace` does afterwards — including a `replace` that swaps in a different object or that runs while a reader holds only the copy. `replace` itself builds `next` outside the lock (an allocation under no contention) then swaps under the lock; no defect. Under the stated contract ("snapshot remains valid after unlock and later replacement") `read` is correct as written; the only hazard would have been returning `std::string_view`/raw pointer into `value_`, which it does not. No finding to report here — consistent with the instruction to report only contract-justified defects.

## Test plan (short)

- `copy_range`: wrap case above → false; `offset==size, length==0` → true; normal copy → true and bytes match; `length > size` → false.
- `display_name`: call twice, use the first view after the second call (per contract) — readable both times; content `"worker-7"`.
- `Cache`: TSan/TSanO race repro of concurrent put/get → clean under the locked version; get-after-put returns the value; put-after-get keeps an earlier snapshot usable.
- `make_tasks`: build 4 tasks, invoke all after `make_tasks` returned → outputs are exactly {0,1,2,3} once each, no dangling read (TSan clean).
- `valid_index`: `(0,0)`→false, `(5,5)`→false, `(5,3)`→true.
- `Snapshot`: reader holds `read()` across a concurrent `replace`, then reads the string → old value intact; replace/replace swap works.

All five defects are in the target file; Snapshot contributes none.