// GLM's independent-sequence scheduler. Included after the generation helpers so the
// single-request path uses the same verified MTP implementation as ordinary serving.
// --prefix-cache=K: every slot keeps its own prompt tokens and K checkpoints (src/program/glm_prefix_cache.hpp), so
// a conversation that comes back to its slot resumes there.  BGEN / GEN take the single-stream keys: pin=N, ckpt=0,
// temperature= top_p= top_k= min_p= seed= (per slot; a token is drawn at its absolute position).
static int serve_sequences(Decoder &decoder, int count, int context, int width,
                           const std::vector<int> &stops, bool mtp, int depth, int prefix_entries = 0,
                           ServeStateCache *state = nullptr) {
    const char *spec = std::getenv("STRATA_GLM_BATCH_MTP");
    const bool batch_mtp = mtp && spec && std::string(spec) == "1";
    if (batch_mtp && depth > 2) throw std::invalid_argument("GLM: batch MTP supports depth 1 or 2");
    using Clock = std::chrono::steady_clock;
    auto ms = [](auto begin) { return std::chrono::duration<double, std::milli>(Clock::now() - begin).count(); };
    struct Slot {
        bool active = false;
        int token = 0, generated = 0, limit = 0;
        Clock::time_point start;
        strata::spec::HostSampler sampler;   // greedy unless the request's keys set a temperature
        int64_t next = 0;                    // the absolute position of the next token drawn
    };
    std::vector<strata::glm::PrefixCache<Decoder::HostCheckpoint>> prefixes;
    for (int i = 0; i < count; ++i) prefixes.emplace_back(size_t(prefix_entries));
    auto draw = [&](Slot &s, const std::vector<float> &logits, size_t offset, int64_t position) {
        if (s.sampler.greedy()) return greedy(logits, offset, decoder.vocabulary());
        return s.sampler.pick(logits.data() + offset, decoder.vocabulary(), uint64_t(position));
    };
    struct Request { std::string line; std::shared_ptr<std::atomic<bool>> cancel; };
    std::vector<Slot> slots(count);
    std::array<std::atomic<bool>, 8> stopped{}, yielded{};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Request> requests;
    std::shared_ptr<std::atomic<bool>> active;
    bool ended = false;
    std::jthread reader([&] {
        for (std::string line; std::getline(std::cin, line);) {
            std::lock_guard lock(mutex);
            if (line == "STOP") {
                if (active) *active = true;
                else if (!requests.empty()) *requests.front().cancel = true;
            } else if (line.starts_with("BSTOP ") || line.starts_with("BYIELD ")) {
                std::istringstream in(line);
                std::string op, extra; int slot;
                if (in >> op >> slot && !(in >> extra) && slot >= 0 && slot < count) {
                    if (op == "BSTOP") stopped[slot] = true;
                    else yielded[slot] = true;
                }
            } else {
                if (line.starts_with("BGEN ")) {
                    std::istringstream in(line); std::string op; int slot;
                    if (in >> op >> slot && slot >= 0 && slot < count) {
                        stopped[slot] = false; yielded[slot] = false;
                    }
                }
                const bool quit = line == "QUIT";
                requests.push_back({std::move(line), std::make_shared<std::atomic<bool>>(false)});
                ready.notify_one();
                if (quit) break;
            }
        }
        std::lock_guard lock(mutex);
        ended = true; ready.notify_one();
    });
    auto is_stop = [&](int token) { return std::find(stops.begin(), stops.end(), token) != stops.end(); };
    auto running = [&] { return std::any_of(slots.begin(), slots.end(), [](const Slot &s) { return s.active; }); };
    auto finish = [&](int slot, const char *reason) {
        auto &s = slots[slot];
        std::cout << "BDONE " << slot << ' ' << s.generated << ' ' << reason << ' ' << ms(s.start) << '\n' << std::flush;
        s.active = false;
    };
    auto speculative_pump = [&](const std::vector<int> &ids) {
        const auto start = Clock::now();
        int delivered = 0, proposed = 0, accepted_total = 0, replayed = 0;
        for (size_t begin = 0; begin < ids.size();) {
            struct Work { int slot, offset, position; std::vector<int> window; Decoder::Checkpoint checkpoint; };
            std::vector<Work> work;
            std::vector<int> rows, input;
            decoder.begin_sequence_window();
            try {
                while (begin < ids.size()) {
                    const int id = ids[begin]; auto &slot = slots[id];
                    decoder.select_sequence(id);
                    // A sequence without a primed draft state (never reached prepare_mtp) verifies only its token.
                    const int k = decoder.draft_ready() ? std::min(depth, slot.limit - slot.generated - 1) : 0;
                    if (input.size() + k + 1 > 8) break;
                    work.push_back({id, (int)input.size(), decoder.token_position(), {slot.token}, {}});
                    // row a of this window draws position token_position() + a + 1 (slot.next before the round)
                    auto &w = work.back();
                    // A one-row tail cannot reject a draft and needs no rollback allocation.
                    // In particular, eight one-token tails must not allocate eight full checkpoints.
                    if (k) { w.checkpoint = decoder.checkpoint(); decoder.save(w.checkpoint); }
                    for (int d = 0; d < k; ++d) w.window.push_back(greedy(decoder.mtp_propose(w.window.back(), d == 0)));
                    proposed += k;
                    rows.insert(rows.end(), w.window.size(), id);
                    input.insert(input.end(), w.window.begin(), w.window.end());
                    ++begin;
                }
                auto logits = decoder.step_sequences(rows, input, true);
                for (auto &w : work) {
                    auto &slot = slots[w.slot];
                    int accepted = 0;
                    auto emit = [&](int token) {
                        slot.token = token; ++slot.generated; ++delivered;
                        std::cout << "BT " << w.slot << ' ' << token << '\n';
                        if (is_stop(token)) finish(w.slot, "stop");
                        else if (slot.generated == slot.limit) finish(w.slot, "length");
                    };
                    while (accepted + 1 < (int)w.window.size() &&
                           w.window[accepted + 1] == draw(slot, logits, (size_t)(w.offset + accepted) * decoder.vocabulary(),
                                                          w.position + accepted + 1)) {
                        emit(w.window[++accepted]);
                        if (!slot.active) break;
                    }
                    accepted_total += accepted;
                    if (!slot.active) continue;
                    const int next = draw(slot, logits, (size_t)(w.offset + accepted) * decoder.vocabulary(),
                                          w.position + accepted + 1);
                    decoder.select_sequence(w.slot);
                    const int verified = w.window.size();
                    w.window.resize(accepted + 1);
                    if (accepted + 1 != verified) {
                        decoder.restore(w.checkpoint);
                        decoder.batch(w.window, false);
                        replayed += w.window.size();
                    }
                    if (w.checkpoint.storage) decoder.restore_mtp(w.checkpoint);
                    decoder.sync_mtp(w.window, w.position);
                    emit(next);
                }
            } catch (...) { decoder.end_sequence_window(); throw; }
            decoder.end_sequence_window();
        }
        std::cout << std::flush;
        std::cerr << "BATCH_DECODE active=" << ids.size() << " generated=" << delivered << " ms=" << ms(start) << '\n'
                  << "BATCH_MTP proposed=" << proposed << " accepted=" << accepted_total << " replayed=" << replayed << '\n';
    };
    auto pump = [&] {
        std::vector<int> ids, tokens;
        for (int i = 0; i < count; ++i) if (slots[i].active) {
            if (stopped[i].exchange(false)) finish(i, "cancel");
            else { ids.push_back(i); tokens.push_back(slots[i].token); }
        }
        if (ids.empty()) return;
        if (batch_mtp) { speculative_pump(ids); return; }
        const auto start = Clock::now();
        auto logits = decoder.step_sequences(ids, tokens);
        for (size_t row = 0; row < ids.size(); ++row) {
            auto &s = slots[ids[row]];
            s.token = draw(s, logits, row * decoder.vocabulary(), s.next++);
            ++s.generated;
            std::cout << "BT " << ids[row] << ' ' << s.token << '\n';
            if (is_stop(s.token)) finish(ids[row], "stop");
            else if (s.generated >= s.limit) finish(ids[row], "length");
        }
        std::cout << std::flush;
        std::cerr << "BATCH_DECODE active=" << ids.size() << " generated=" << ids.size() << " ms=" << ms(start) << '\n';
    };
    auto safe_pump = [&] {
        try { pump(); }
        catch (const std::exception &e) {
            // A failed shared step can have modified every active sequence.
            std::cout << "ERR batch decode failed: " << e.what() << '\n' << std::flush;
            std::_Exit(1);
        }
    };
    std::cout << "INFO engine=glm-experimental sampling=greedy batch_slots=" << count
              << "\nREADY " << context << " stop prefill-includes-reset" << (prefix_entries ? " resume" : "") << '\n'
              << std::flush;
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex);
            if (!running()) ready.wait(lock, [&] { return ended || !requests.empty(); });
            if (!requests.empty()) {
                request = std::move(requests.front()); requests.pop_front(); active = request.cancel;
            } else if (ended && !running()) break;
        }
        if (request.line == "QUIT") {
            for (int i = 0; i < count; ++i) if (slots[i].active) finish(i, "cancel");
            std::cout << std::flush;
            break;
        }
        if (!request.cancel) {          // nothing was queued: decode one step for the running slots
            safe_pump();
            continue;
        }
        if (request.line.empty()) {     // a blank line from the client is a malformed request
            std::cout << "ERR empty request line\n" << std::flush;
            { std::lock_guard lock(mutex); active.reset(); }
            continue;
        }
        bool batched = false;
        int slot = -1, delivered_control = 0;
        try {
            std::istringstream in(request.line);
            std::string op; int limit;
            in >> op;
            batched = op == "BGEN";
            if (batched) in >> slot;
            else if (op == "GEN") slot = 0;
            std::vector<std::string> words;
            if (slot >= 0 && slot < count && (in >> limit)) for (std::string w; in >> w;) words.push_back(w);
            if (slot < 0 || slot >= count || words.empty() || limit < 1 || (batched ? slots[slot].active : running()))
                throw std::invalid_argument("expected GEN <max_new> [key=value ...] <ids> or BGEN <free-slot> <max_new> [key=value ...] <ids>");
            int pin = 0;
            bool save_prompt = true;
            strata::spec::HostSampler sampler;
            uint64_t seed = 0;
            for (size_t i = 0; i + 1 < words.size(); ++i) {
                const auto eq = words[i].find('=');
                const std::string key = words[i].substr(0, eq), value = eq == std::string::npos ? "" : words[i].substr(eq + 1);
                if (value.empty()) throw std::invalid_argument("GLM: GEN key without a value: " + words[i]);
                if (key == "pin" && prefix_entries) pin = std::stoi(value);
                else if (key == "ckpt" && prefix_entries && (value == "0" || value == "1")) save_prompt = value == "1";
                else if (key == "temperature") sampler.temperature = std::stof(value);
                else if (key == "top_p") sampler.top_p = std::stof(value);
                else if (key == "top_k") sampler.top_k = std::stoi(value);
                else if (key == "min_p") sampler.min_p = std::stof(value);
                else if (key == "seed") seed = std::stoull(value);
                else throw std::invalid_argument("GLM: unknown GEN key " + words[i]);
            }
            if (!sampler.greedy())
                sampler.seed = seed ? seed : std::random_device{}() | (uint64_t(std::random_device{}()) << 32);
            const auto prompt = token_ids(words.back());
            if (prompt.empty() || prompt.size() + (size_t)limit > (size_t)context ||
                std::any_of(prompt.begin(), prompt.end(), [&](int t) { return t < 0 || t >= decoder.vocabulary(); }))
                throw std::invalid_argument("request exceeds context or has invalid token counts");
            decoder.select_sequence(slot);
            const bool slot_mtp = mtp && (!batched || batch_mtp);
            auto &cache = prefixes[slot];
            auto *entry = prefix_entries ? cache.choose(prompt) : nullptr;
            if (entry && slot_mtp && !entry->store.draft_at(entry->pos)) entry = nullptr;
            // a stored state shared by every slot (--state-*): a system prompt, a parked conversation, after a restart
            std::shared_ptr<const strata::glm::GlmStateImage> image;
            if (state && state->on()) {
                image = state->store->find(prompt, state->candidates(prompt));
                if (image && (image->position <= (entry ? entry->pos : 0) || (slot_mtp && !image->draft_synced()))) image.reset();
            }
            int resumed = 0;
            if (image) {
                decoder.reset();
                decoder.import_state(*image);
                resumed = image->position;
                std::cerr << "BATCH_STATE_HIT slot=" << slot << " pos=" << resumed << '\n';
                cache.clear();
                cache.held.assign(prompt.begin(), prompt.begin() + resumed);
            } else if (entry) {
                decoder.begin_request(true);
                decoder.restore_host(entry->store);
                resumed = entry->pos;
                cache.truncate(resumed);
            } else {
                decoder.reset();
                cache.clear();
            }
            if (prefix_entries) std::cout << "RESUME " << resumed << '\n' << std::flush;
            request_sampling = {sampler, !sampler.greedy()};    // the solo GEN path's generate_mtp draws with it
            decoder.enable_mtp_capture(slot_mtp);
            const auto start = Clock::now();
            auto cancelled = [&] { return bool(*request.cancel) || (batched && stopped[slot].load()); };
            decoder.set_cancel(cancelled);
            std::vector<float> logits;
            bool did_yield = false;
            const int root_at = state && state->on() ? state->root(prompt) : 0;
            const bool want_root = root_at > resumed && !state->store->contains(prompt, root_at);
            std::shared_ptr<strata::glm::GlmStateImage> root_image;
            for (size_t at = resumed; at < prompt.size() && !cancelled();) {
                size_t end = std::min(prompt.size(), at + width);
                const bool at_pin = prefix_entries && !slot_mtp && pin > (int)at && (size_t)pin < end;
                if (at_pin) end = pin;
                const bool at_root = want_root && root_at > (int)at && (size_t)root_at < end;
                if (at_root) end = root_at;
                logits = decoder.batch(std::vector<int>(prompt.begin() + at, prompt.begin() + end));
                if (at_root) {
                    root_image = std::make_shared<strata::glm::GlmStateImage>();
                    decoder.export_fixed(*root_image);
                    root_image->tokens.assign(prompt.begin(), prompt.begin() + root_at);
                }
                if (at_pin) {
                    cache.held.assign(prompt.begin(), prompt.begin() + end);
                    if (auto *e = cache.slot_for(pin, true)) { decoder.save_host(e->store); cache.commit(e); }
                }
                at = end;
                std::cout << "PP " << end << ' ' << prompt.size() << '\n' << std::flush;
                if (batched && yielded[slot].exchange(false) && end < prompt.size()) {
                    std::cout << "YIELDED " << slot << ' ' << end << '\n'; did_yield = true; break;
                }
                // A long admission allows already admitted sequences a decode step per chunk.
                decoder.set_cancel({}); if (!batch_mtp) safe_pump(); decoder.set_cancel(cancelled);
                decoder.select_sequence(slot);         // the pump may have selected another slot
            }
            auto prefilled = Clock::now();
            int generated = 0;
            bool cont = false;
            std::string reason = cancelled() ? "cancel" : did_yield ? "yield" : "stop";
            const bool prompt_done = !cancelled() && !did_yield;
            if (prefix_entries && prompt_done) cache.held = prompt;
            else if (prefix_entries) cache.clear();
            if (prompt_done) {
                if (slot_mtp) { decoder.prepare_mtp(prompt, resumed); if (!batched) decoder.prepare_mtp_history(depth); }
                if (prefix_entries && save_prompt)
                    if (auto *e = cache.slot_for((int)prompt.size(), false)) { decoder.save_host(e->store); cache.commit(e); }
                if (state && state->on()) {
                    decoder.select_sequence(slot);
                    if (root_image) {
                        decoder.export_rows(*root_image);
                        state->store->put(root_image);
                        std::cerr << "STATE_SAVE kind=root slot=" << slot << " pos=" << root_image->position << '\n';
                    }
                    if ((int)prompt.size() >= state->root_min && save_prompt) {
                        auto turn = std::make_shared<strata::glm::GlmStateImage>();
                        decoder.export_fixed(*turn);
                        decoder.export_rows(*turn);
                        turn->tokens = prompt;
                        state->store->put(turn);
                        const auto st = state->store->stats();
                        std::cerr << "STATE_SAVE kind=turn slot=" << slot << " pos=" << turn->position << " ram_hits="
                                  << st.ram_hits << " disk_hits=" << st.disk_hits << " misses=" << st.misses << '\n';
                    }
                }
                decoder.prepare_decode_cache(); decoder.freeze_cache(); decoder.prepare_cpu(0);
                prefilled = Clock::now();
                if (batched) {
                    Slot admitted{false, 0, 1, limit, Clock::now(), sampler, (int64_t)prompt.size()};
                    const int token = draw(admitted, logits, 0, admitted.next++);
                    std::cout << "T " << token << '\n'; generated = delivered_control = 1;
                    cont = !is_stop(token) && limit > 1;
                    admitted.active = cont; admitted.token = token;
                    slots[slot] = admitted;
                    reason = is_stop(token) ? "stop" : "length";
                } else if (mtp) {
                    auto result = generate_mtp(decoder, logits, limit, stops, depth, [&](int token) {
                        if (cancelled()) throw std::runtime_error("cancelled");
                        ++delivered_control;
                        std::cout << "T " << token << '\n' << std::flush;
                    }, &prompt);
                    generated = result.generated; reason = result.stopped ? "stop" : "length";
                } else {
                    reason = "length";
                    for (; generated < limit && !cancelled();) {
                        const int token = pick(logits, decoder.token_position());
                        ++generated; ++delivered_control; std::cout << "T " << token << '\n' << std::flush;
                        if (is_stop(token)) { reason = "stop"; break; }
                        if (generated < limit) logits = decoder.step(token);
                    }
                    if (cancelled()) reason = "cancel";
                }
            }
            std::cout << "DONE " << generated << ' ' << prompt.size() << ' ' << std::chrono::duration<double, std::milli>(prefilled - start).count()
                      << ' ' << ms(prefilled) << ' ' << reason << " 0 0 " << resumed << '\n';
            if (batched) std::cout << "BADM " << slot << ' ' << int(cont) << '\n';
            std::cout << std::flush;
        } catch (const std::exception &e) {
            decoder.set_cancel({});
            if ((*request.cancel || (batched && slot >= 0 && slot < count && stopped[slot])) &&
                std::string(e.what()) == "cancelled") {
                std::cout << "DONE " << delivered_control << " 0 0 0 cancel 0 0 0\n";
                if (batched) std::cout << "BADM " << slot << " 0\n";
                std::cout << std::flush;
            } else {
                std::cout << "ERR " << e.what() << '\n' << std::flush;
            }
            if (slot >= 0 && slot < count) prefixes[slot].clear();
        }
        decoder.set_cancel({});
        { std::lock_guard lock(mutex); active.reset(); }
    }
    decoder.set_cancel({}); decoder.report_gpu(); decoder.report_cache();
    return 0;
}
