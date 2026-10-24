// Batched-decode CPU expert traffic estimate from single-stream native FP4 routing traces.
//
// Input: JSON-lines traces written by strata-glm-decode with --routing-trace=PATH, STRATA_GLM_ROUTING_EVENTS=1 and
// STRATA_GLM_ROUTING_WEIGHTS=1 or 2 (weights; 2 adds router candidates), one or more requests per file. Decode rows
// are one-row "target" events (plain decode) or "verify" windows (MTP), grouped per request; the kept experts come
// from the {"kind":"fp4_keep"} events. Independent requests are lined up as the slots of a batch, step by step, and
// each step's distinct CPU experts per layer are counted under a keep-set and route-drop policy. The drop rules are
// the engine's own (include/strata/host/glm_route_drop.h). This is a traffic model, not a timing: it reads no
// weights and ignores GPU time, admissions and caches.
//
//   strata-glm-batch-route-sim [options] TRACE...
//     --batch=1,8,16,32   slots per step            --keep=own|first|prior|oracle (comma list)
//     --drop=none|token|batch (comma list)           --fraction=0.5 --max-drop=7 --cap=0.5
//     --state-gb=0,4,8    GPU 0 bytes taken by slot state (shrinks GPU 0's keep sets)
//     --affinity=0.025    route again from the candidates with this bonus for kept experts (needs mode 2 traces)
//     --expert-mb=14.16   bytes read per CPU expert pass      --bandwidth=70,85,100,118 (GB/s ceilings)
//     --prior=TRACE,...   build the prior keep sets from these traces instead of leaving the batch's own requests
//                         out of the traces under test (needed when a batch uses most of them)
//     --write-prior=PATH  write the traces' decode route counts as an engine keep prior (STRATA_GLM_FP4_BATCH_KEEP=
//                         prior:PATH): one line per layer, "layer count0 ... count287"
//     --json=PATH         also write every result row         --self-test
#include "strata/host/glm_route_drop.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A small JSON reader: numbers, strings, arrays, objects; enough for the trace events.
struct Json {
    enum Kind { Null, Number, String, Array, Object } kind = Null;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;
    const Json *get(const std::string &key) const {
        for (const auto &[k, v] : fields) if (k == key) return &v;
        return nullptr;
    }
};
struct Parser {
    const std::string &s;
    size_t i = 0;
    void space() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) ++i; }
    [[noreturn]] void fail() const { throw std::runtime_error("trace: malformed JSON near offset " + std::to_string(i)); }
    std::string string() {
        if (s[i] != '"') fail();
        std::string out;
        for (++i; i < s.size() && s[i] != '"'; ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) ++i;
            out += s[i];
        }
        if (i >= s.size()) fail();
        ++i;
        return out;
    }
    Json value() {
        space();
        if (i >= s.size()) fail();
        Json v;
        if (s[i] == '{') {
            v.kind = Json::Object; ++i; space();
            if (s[i] == '}') { ++i; return v; }
            for (;;) {
                space(); std::string key = string(); space();
                if (s[i] != ':') fail();
                ++i; v.fields.emplace_back(std::move(key), value()); space();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return v; }
                fail();
            }
        }
        if (s[i] == '[') {
            v.kind = Json::Array; ++i; space();
            if (s[i] == ']') { ++i; return v; }
            for (;;) {
                v.items.push_back(value()); space();
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == ']') { ++i; return v; }
                fail();
            }
        }
        if (s[i] == '"') { v.kind = Json::String; v.text = string(); return v; }
        if (s.compare(i, 4, "null") == 0) { i += 4; return v; }
        char *end = nullptr;
        v.number = std::strtod(s.c_str() + i, &end);
        if (end == s.c_str() + i) fail();
        v.kind = Json::Number; i = size_t(end - s.c_str());
        return v;
    }
};

constexpr int kMainLayers = 45;   // routed layers 0..44; the MTP draft layer (45) runs on the GPU

struct Row {
    std::array<int, 8> id{};
    std::array<float, 8> weight{};
    std::vector<std::array<float, 3>> candidates;   // [id, sigmoid, unbonused score], best first
};
struct Step {
    int position = 0, advances = 1;
    std::map<int, std::vector<Row>> layers;   // layer -> the step's rows (one, or an MTP verify window)
};
struct Request {
    std::string name;
    std::vector<Step> steps;
    std::map<std::pair<int, int>, std::set<int>> keep;   // (device, layer) -> kept experts
};

std::vector<Request> load(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::map<int, Request> requests;
    std::map<std::pair<int, long long>, Step> steps;   // (request, round or position) -> step
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '{') continue;
        Parser p{line};
        const Json e = p.value();
        const Json *request = e.get("request");
        if (!request) continue;
        const int r = int(request->number);
        if (const Json *kind = e.get("kind")) {
            if (kind->text == "fp4_keep") {
                auto &set = requests[r].keep[{int(e.get("device")->number), int(e.get("layer")->number)}];
                set.clear();
                for (const auto &x : e.get("experts")->items) set.insert(int(x.number));
            }
            continue;
        }
        const Json *routes = e.get("routes"), *phase = e.get("phase");
        if (!routes || !phase) continue;
        const int layer = int(e.get("layer")->number), position = int(e.get("position")->number);
        const int round = int(e.get("round")->number);
        const size_t nt = routes->items.size();
        if (layer < 0 || layer >= kMainLayers) continue;
        long long key;
        if (phase->text == "verify") key = round;
        else if (phase->text == "target" && round < 0 && nt == 1) key = (1ll << 40) + position;
        else continue;   // prefill, drafts, resynchronization and probes
        Step &step = steps[{r, key}];
        step.position = position;
        auto &rows = step.layers[layer];
        rows.clear();
        const Json *weights = e.get("weights"), *candidates = e.get("candidates");
        for (size_t t = 0; t < nt; ++t) {
            Row row;
            for (int j = 0; j < 8; ++j) {
                row.id[size_t(j)] = int(routes->items[t].items.at(size_t(j)).number);
                row.weight[size_t(j)] = weights ? float(weights->items[t].items.at(size_t(j)).number) : 1.f;
            }
            if (candidates)
                for (const auto &c : candidates->items[t].items)
                    row.candidates.push_back({float(c.items.at(0).number), float(c.items.at(1).number),
                                              float(c.items.at(2).number)});
            rows.push_back(std::move(row));
        }
    }
    for (auto &[key, step] : steps) requests[key.first].steps.push_back(std::move(step));
    std::vector<Request> out;
    for (auto &[r, request] : requests) {
        if (request.steps.empty()) continue;
        std::sort(request.steps.begin(), request.steps.end(),
                  [](const Step &a, const Step &b) { return a.position < b.position; });
        for (size_t s = 0; s + 1 < request.steps.size(); ++s)
            request.steps[s].advances = std::max(1, request.steps[s + 1].position - request.steps[s].position);
        request.name = path + "#" + std::to_string(r);
        out.push_back(std::move(request));
    }
    return out;
}

struct Options {
    std::vector<int> batches{1, 8, 16, 32};
    std::vector<std::string> keeps{"own", "first", "prior", "oracle"}, drops{"token", "batch"};
    std::vector<double> state_gb{0}, bandwidth{70, 85, 100, 118};
    double fraction = 0.5, cap = 0.5, affinity = 0, expert_bytes = 14.16e6;
    unsigned max_drop = 7;
    std::string json;
    std::vector<std::string> prior;
    std::string write_prior;
};

using KeepSets = std::map<std::pair<int, int>, std::set<int>>;

// Median keep size per (device, layer) over the requests' keep events.
std::map<std::pair<int, int>, size_t> keep_sizes(const std::vector<Request> &requests) {
    std::map<std::pair<int, int>, std::vector<size_t>> sizes;
    for (const auto &r : requests) for (const auto &[key, set] : r.keep) sizes[key].push_back(set.size());
    std::map<std::pair<int, int>, size_t> out;
    for (auto &[key, v] : sizes) { std::sort(v.begin(), v.end()); out[key] = v[v.size() / 2]; }
    return out;
}

using Counts = std::map<int, std::vector<double>>;   // layer -> route counts per expert
void add_counts(Counts &counts, const Request &r, double sign) {
    for (const auto &step : r.steps)
        for (const auto &[layer, rows] : step.layers) {
            auto &c = counts[layer];
            if (c.empty()) c.assign(512, 0);
            for (const auto &row : rows) for (int id : row.id) if (id >= 0 && id < 512) c[size_t(id)] += sign;
        }
}

// The engine's parity ownership: each device keeps the top experts of its parity by count.
KeepSets select_keep(const Counts &counts, const std::map<std::pair<int, int>, size_t> &sizes, size_t shrink0) {
    KeepSets keep;
    for (const auto &[key, size] : sizes) {
        const auto [device, layer] = key;
        const auto it = counts.find(layer);
        size_t count = size;
        if (device == 0) count = count > shrink0 ? count - shrink0 : 0;
        std::vector<std::pair<double, int>> order;
        if (it != counts.end())
            for (int id = device; id < int(it->second.size()); id += 2) order.push_back({-it->second[size_t(id)], id});
        std::sort(order.begin(), order.end());
        auto &set = keep[key];
        for (size_t i = 0; i < order.size() && i < count; ++i) if (order[i].first < 0) set.insert(order[i].second);
    }
    return keep;
}

bool kept(const KeepSets &keep, int layer, int id) {
    const auto it = keep.find({id & 1, layer});
    return it != keep.end() && it->second.count(id);
}

// Route a row again under a keep set's affinity bonus, as the engine's router does (top 8 by score + bonus).
Row reroute(const Row &row, const KeepSets &keep, int layer, double affinity) {
    if (affinity <= 0 || row.candidates.size() < 8) return row;
    std::vector<std::pair<double, int>> order;
    for (size_t j = 0; j < row.candidates.size(); ++j) {
        const int id = int(row.candidates[j][0]);
        order.push_back({-(row.candidates[j][2] + (kept(keep, layer, id) ? affinity : 0.0)), int(j)});
    }
    std::sort(order.begin(), order.end(), [&](const auto &a, const auto &b) {
        return a.first < b.first || (a.first == b.first && row.candidates[size_t(a.second)][0] < row.candidates[size_t(b.second)][0]);
    });
    Row out;
    for (int j = 0; j < 8; ++j) {
        const auto &c = row.candidates[size_t(order[size_t(j)].second)];
        out.id[size_t(j)] = int(c[0]); out.weight[size_t(j)] = c[1];
    }
    return out;
}

struct Result {
    int batch = 0;
    std::string keep, drop;
    double state_gb = 0;
    double steps = 0, tokens = 0, rows = 0, cpu_groups = 0, gpu_groups = 0, cpu_routes = 0, wide_cpu_groups = 0;
    double dropped_share_sum = 0, dropped_share_max = 0;
    std::vector<double> dropped_shares;
    int max_cpu_rows = 0, max_gpu_rows = 0, layers = 0;
};

// One batch step: rows of every slot at one layer.
void count_layer(const std::vector<Row> &rows, const KeepSets &keep, int layer, const Options &o,
                 const std::string &drop, Result &result) {
    const size_t n = rows.size();
    std::vector<uint8_t> eligible(n, 0), masks(n, 0);
    std::vector<double> shares(n, 0);
    std::vector<float> weights(n * 8);
    std::vector<int> ids(n * 8);
    for (size_t r = 0; r < n; ++r)
        for (int j = 0; j < 8; ++j) {
            weights[r * 8 + size_t(j)] = rows[r].weight[size_t(j)];
            ids[r * 8 + size_t(j)] = rows[r].id[size_t(j)];
            if (!kept(keep, layer, rows[r].id[size_t(j)])) eligible[r] |= uint8_t(1u << j);
        }
    if (drop == "token")
        for (size_t r = 0; r < n; ++r)
            masks[r] = strata_glm_route_drop_capped(weights.data() + r * 8, eligible[r], o.fraction, o.max_drop, o.cap,
                                                    &shares[r]);
    else if (drop == "batch") {
        if (n > STRATA_GLM_ROUTE_DROP_BATCH_ROWS) throw std::runtime_error("batch drop: too many rows in one step");
        strata_glm_route_drop_batch(weights.data(), ids.data(), eligible.data(), unsigned(n), o.fraction, o.max_drop,
                                    o.cap, masks.data(), shares.data());
    } else if (drop != "none") throw std::invalid_argument("--drop must be none, token or batch");
    std::map<int, int> cpu, gpu;
    for (size_t r = 0; r < n; ++r)
        for (int j = 0; j < 8; ++j) {
            if (masks[r] & (1u << j)) continue;
            if (eligible[r] & (1u << j)) ++cpu[rows[r].id[size_t(j)]], ++result.cpu_routes;
            else ++gpu[rows[r].id[size_t(j)]];
        }
    result.cpu_groups += double(cpu.size());
    result.gpu_groups += double(gpu.size());
    for (const auto &[id, count] : cpu) {
        if (count > 8) ++result.wide_cpu_groups;
        result.max_cpu_rows = std::max(result.max_cpu_rows, count);
    }
    for (const auto &[id, count] : gpu) result.max_gpu_rows = std::max(result.max_gpu_rows, count);
    for (size_t r = 0; r < n; ++r) {
        result.dropped_share_sum += shares[r];
        result.dropped_share_max = std::max(result.dropped_share_max, shares[r]);
        result.dropped_shares.push_back(shares[r]);
    }
    ++result.layers;
}

Result simulate(const std::vector<Request> &requests, int batch, const std::string &keep_policy,
                const std::string &drop, double state_gb, const Options &o, const Counts *prior = nullptr) {
    Result result;
    result.batch = batch; result.keep = keep_policy; result.drop = drop; result.state_gb = state_gb;
    const auto sizes = keep_sizes(requests);
    const size_t shrink0 = size_t(state_gb * 1e9 / (42.0 * o.expert_bytes));
    Counts total;
    for (const auto &r : requests) add_counts(total, r, 1);
    const size_t n = requests.size();
    const size_t groups = std::max<size_t>(1, n / size_t(batch));
    for (size_t g = 0; g < groups; ++g) {
        std::vector<size_t> slots;
        for (int i = 0; i < batch; ++i) slots.push_back((g * size_t(batch) + size_t(i)) % n);
        KeepSets shared;
        if (keep_policy == "first") shared = requests[slots[0]].keep;
        else if (keep_policy == "prior" || keep_policy == "oracle") {
            Counts counts;
            if (keep_policy == "prior" && prior) counts = *prior;
            else if (keep_policy == "prior") {
                counts = total;
                std::set<size_t> unique(slots.begin(), slots.end());
                for (size_t s : unique) add_counts(counts, requests[s], -1);
            } else for (size_t s : slots) add_counts(counts, requests[s], 1);
            shared = select_keep(counts, sizes, shrink0);
        } else if (keep_policy != "own") throw std::invalid_argument("--keep must be own, first, prior or oracle");
        size_t length = SIZE_MAX;
        for (size_t s : slots) length = std::min(length, requests[s].steps.size());
        for (size_t step = 0; step < length; ++step) {
            std::map<int, std::vector<Row>> layers;
            for (size_t i = 0; i < slots.size(); ++i) {
                const Request &r = requests[slots[i]];
                // A repeated request (more slots than traces) starts at another step, so its rows differ.
                const size_t at = (step + (i / n) * (length / 2 + 1)) % r.steps.size();
                const Step &s = r.steps[at];
                result.tokens += s.advances;
                for (const auto &[layer, rows] : s.layers) {
                    const KeepSets &keep = keep_policy == "own" ? r.keep : shared;
                    for (const auto &row : rows) layers[layer].push_back(reroute(row, keep, layer, o.affinity));
                    result.rows += double(rows.size());
                }
            }
            for (const auto &[layer, rows] : layers) {
                // Own keep sets differ per slot; each slot's rows then count against its own set.
                if (keep_policy == "own") {
                    size_t begin = 0;
                    for (size_t i = 0; i < slots.size(); ++i) {
                        const Step &s = requests[slots[i]].steps[(step + (i / n) * (length / 2 + 1)) % requests[slots[i]].steps.size()];
                        const auto it = s.layers.find(layer);
                        const size_t count = it == s.layers.end() ? 0 : it->second.size();
                        std::vector<Row> own(rows.begin() + long(begin), rows.begin() + long(begin + count));
                        if (!own.empty()) count_layer(own, requests[slots[i]].keep, layer, o, drop, result);
                        begin += count;
                    }
                } else count_layer(rows, shared, layer, o, drop, result);
            }
            ++result.steps;
        }
    }
    return result;
}

std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> out; std::stringstream in(s); std::string item;
    while (std::getline(in, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

void print(const Result &r, const Options &o, std::ostream &out, bool json) {
    const double steps = std::max(1.0, r.steps);
    const double bytes = r.cpu_groups * o.expert_bytes / steps;
    const double per_token = r.cpu_groups * o.expert_bytes / std::max(1.0, r.tokens);
    std::vector<double> shares = r.dropped_shares;
    std::sort(shares.begin(), shares.end());
    const double p99 = shares.empty() ? 0 : shares[std::min(shares.size() - 1, size_t(0.99 * double(shares.size())))];
    const double layers = std::max(1.0, r.steps) * 42.0;
    if (json) {
        out << "{\"batch\":" << r.batch << ",\"keep\":\"" << r.keep << "\",\"drop\":\"" << r.drop
            << "\",\"state_gb\":" << r.state_gb << ",\"steps\":" << r.steps << ",\"tokens_per_step\":" << r.tokens / steps
            << ",\"rows_per_step\":" << r.rows / steps / 42.0 << ",\"cpu_experts_per_layer\":" << r.cpu_groups / layers
            << ",\"gpu_experts_per_layer\":" << r.gpu_groups / layers << ",\"cpu_routes_per_layer\":" << r.cpu_routes / layers
            << ",\"cpu_gb_per_step\":" << bytes / 1e9 << ",\"cpu_gb_per_token\":" << per_token / 1e9
            << ",\"cpu_groups_over_8_rows\":" << r.wide_cpu_groups << ",\"max_cpu_rows\":" << r.max_cpu_rows
            << ",\"max_gpu_rows\":" << r.max_gpu_rows << ",\"dropped_share_mean\":"
            << r.dropped_share_sum / std::max<size_t>(1, r.dropped_shares.size()) << ",\"dropped_share_p99\":" << p99
            << ",\"dropped_share_max\":" << r.dropped_share_max << ",\"ceilings\":{";
        for (size_t i = 0; i < o.bandwidth.size(); ++i)
            out << (i ? "," : "") << '"' << o.bandwidth[i] << "\":" << o.bandwidth[i] * 1e9 / std::max(1.0, per_token);
        out << "}}\n";
        return;
    }
    char line[512];
    std::snprintf(line, sizeof line, "B=%-3d keep=%-6s drop=%-5s state=%4.1fGB steps=%-6.0f tok/step=%6.2f "
                  "cpu_exp/layer=%6.2f gpu_exp/layer=%6.2f GB/step=%6.2f GB/tok=%5.3f >8rows=%-5.0f maxrows=%d/%d "
                  "drop mean/p99/max=%.3f/%.3f/%.3f", r.batch, r.keep.c_str(), r.drop.c_str(), r.state_gb, r.steps,
                  r.tokens / steps, r.cpu_groups / layers, r.gpu_groups / layers, bytes / 1e9, per_token / 1e9,
                  r.wide_cpu_groups, r.max_cpu_rows, r.max_gpu_rows,
                  r.dropped_share_sum / std::max<size_t>(1, r.dropped_shares.size()), p99, r.dropped_share_max);
    out << line;
    for (double bw : o.bandwidth) {
        std::snprintf(line, sizeof line, " %g:%.0f", bw, bw * 1e9 / std::max(1.0, per_token));
        out << line;
    }
    out << '\n';
}

int self_test() {
    // Two requests, two one-row steps each, one layer; device 0 keeps expert 0, device 1 keeps expert 1.
    auto make = [](int a) {
        Request r;
        r.keep[{0, 3}] = {0}; r.keep[{1, 3}] = {1};
        for (int s = 0; s < 2; ++s) {
            Step step; step.position = s;
            Row row;
            for (int j = 0; j < 8; ++j) { row.id[size_t(j)] = j == 0 ? 0 : a + j; row.weight[size_t(j)] = float(j + 1); }
            step.layers[3].push_back(row);
            r.steps.push_back(step);
        }
        return r;
    };
    std::vector<Request> requests{make(10), make(10)};
    Options o;
    // One slot, no drop: 7 CPU experts per step (expert 0 is kept on device 0).
    Result one = simulate(requests, 1, "own", "none", 0, o);
    if (one.steps != 4 || one.cpu_groups != 28 || one.gpu_groups != 4) return 1;
    // The per-token drop at B=1 equals the batch rule at B=1.
    Result token = simulate(requests, 1, "own", "token", 0, o), batch = simulate(requests, 1, "own", "batch", 0, o);
    if (token.cpu_groups != batch.cpu_groups || token.dropped_share_sum != batch.dropped_share_sum) return 2;
    // Two slots routing to the same experts read each CPU expert once per step.
    Result two = simulate(requests, 2, "first", "none", 0, o);
    if (two.steps != 2 || two.cpu_groups != 14 || two.tokens != 4) return 3;
    // The oracle keeps the most routed expert of each parity (0 and 11: 6 CPU experts per step); shrinking GPU 0 by
    // one expert per layer moves expert 0 to the CPU.
    o.expert_bytes = 1e9 / 42.0;
    Result full = simulate(requests, 2, "oracle", "none", 0, o), shrunk = simulate(requests, 2, "oracle", "none", 1, o);
    if (full.cpu_groups != 12 || shrunk.cpu_groups != 14) return 4;
    std::cout << "glm_batch_route_sim self-test passed\n";
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Options o;
        std::vector<std::string> paths;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&](const char *name) -> const char * {
                const std::string prefix = std::string(name) + "=";
                return a.rfind(prefix, 0) == 0 ? argv[i] + prefix.size() : nullptr;
            };
            auto numbers = [](const char *v) { std::vector<double> out; for (const auto &x : split(v)) out.push_back(std::stod(x)); return out; };
            if (a == "--self-test") return self_test();
            if (const char *v = value("--batch")) { o.batches.clear(); for (double x : numbers(v)) o.batches.push_back(int(x)); }
            else if (const char *v = value("--keep")) o.keeps = split(v);
            else if (const char *v = value("--drop")) o.drops = split(v);
            else if (const char *v = value("--state-gb")) o.state_gb = numbers(v);
            else if (const char *v = value("--bandwidth")) o.bandwidth = numbers(v);
            else if (const char *v = value("--fraction")) o.fraction = std::stod(v);
            else if (const char *v = value("--cap")) o.cap = std::stod(v);
            else if (const char *v = value("--max-drop")) o.max_drop = unsigned(std::stoul(v));
            else if (const char *v = value("--affinity")) o.affinity = std::stod(v);
            else if (const char *v = value("--expert-mb")) o.expert_bytes = std::stod(v) * 1e6;
            else if (const char *v = value("--json")) o.json = v;
            else if (const char *v = value("--prior")) o.prior = split(v);
            else if (const char *v = value("--write-prior")) o.write_prior = v;
            else if (a.rfind("--", 0) == 0) throw std::invalid_argument("unknown option " + a);
            else paths.push_back(a);
        }
        if (paths.empty()) throw std::invalid_argument("no traces (see the header of tools/glm_batch_route_sim.cpp)");
        std::vector<Request> requests;
        for (const auto &p : paths) for (auto &r : load(p)) requests.push_back(std::move(r));
        if (requests.empty()) throw std::runtime_error("no decode steps in the traces");
        std::cout << "requests=" << requests.size() << " steps=";
        for (const auto &r : requests) std::cout << r.steps.size() << ' ';
        std::cout << '\n';
        if (!o.write_prior.empty()) {
            Counts counts;
            for (const auto &r : requests) add_counts(counts, r, 1);
            std::ofstream out(o.write_prior);
            out << "# decode route counts from " << requests.size() << " requests (glm_batch_route_sim)\n";
            for (const auto &[layer, c] : counts) {
                out << layer;
                for (int id = 0; id < 288; ++id) out << ' ' << (unsigned long long)std::llround(c[size_t(id)]);
                out << '\n';
            }
            if (!out) throw std::runtime_error("cannot write " + o.write_prior);
            std::cout << "prior written: " << counts.size() << " layers -> " << o.write_prior << '\n';
            if (o.batches.empty()) return 0;
        }
        std::unique_ptr<Counts> prior;
        if (!o.prior.empty()) {
            prior = std::make_unique<Counts>();
            for (const auto &p : o.prior) for (const auto &r : load(p)) add_counts(*prior, r, 1);
        }
        std::unique_ptr<std::ofstream> json;
        if (!o.json.empty()) json = std::make_unique<std::ofstream>(o.json);
        for (int b : o.batches)
            for (const auto &k : o.keeps)
                for (const auto &d : o.drops)
                    for (double gb : o.state_gb) {
                        if (k == "own" && (b > 1 || gb > 0)) continue;   // a shared batch has one keep set
                        const Result r = simulate(requests, b, k, d, gb, o, prior.get());
                        print(r, o, std::cout, false);
                        if (json) print(r, o, *json, true);
                    }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "glm_batch_route_sim: " << e.what() << '\n';
        return 1;
    }
}
