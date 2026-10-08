// GLM's independent-sequence scheduler. Included after the generation helpers so the
// single-request path uses the same verified MTP implementation as ordinary serving.
static int serve_sequences(Decoder &decoder, int count, int context, int width,
                           const std::vector<int> &stops, bool mtp, int depth) {
    const char *spec = std::getenv("STRATA_GLM_BATCH_MTP");
    const bool batch_mtp = mtp && spec && std::string(spec) == "1";
    if (batch_mtp && depth > 2) throw std::invalid_argument("GLM: batch MTP supports depth 1 or 2");
    using Clock = std::chrono::steady_clock;
    auto ms = [](auto begin) { return std::chrono::duration<double, std::milli>(Clock::now() - begin).count(); };
    struct Slot {
        bool active = false;
        int token = 0, generated = 0, limit = 0;
        Clock::time_point start;
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
                    const int k = std::min(depth, slot.limit - slot.generated - 1);
                    if (input.size() + k + 1 > 8) break;
                    decoder.select_sequence(id);
                    work.push_back({id, (int)input.size(), decoder.token_position(), {slot.token}, {}});
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
                           w.window[accepted + 1] == greedy(logits, (size_t)(w.offset + accepted) * decoder.vocabulary(), decoder.vocabulary())) {
                        emit(w.window[++accepted]);
                        if (!slot.active) break;
                    }
                    accepted_total += accepted;
                    if (!slot.active) continue;
                    const int next = greedy(logits, (size_t)(w.offset + accepted) * decoder.vocabulary(), decoder.vocabulary());
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
            auto first = logits.begin() + row * decoder.vocabulary();
            s.token = std::max_element(first, first + decoder.vocabulary()) - first;
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
              << "\nREADY " << context << " stop prefill-includes-reset\n" << std::flush;
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex);
            if (!running()) ready.wait(lock, [&] { return ended || !requests.empty(); });
            if (!requests.empty()) {
                request = std::move(requests.front()); requests.pop_front(); active = request.cancel;
            } else if (ended && !running()) break;
        }
        if (request.line == "QUIT") break;
        if (request.line.empty()) {
            safe_pump();
            continue;
        }
        bool batched = false;
        int slot = -1, delivered_control = 0;
        try {
            std::istringstream in(request.line);
            std::string op, text, extra; int limit;
            in >> op;
            batched = op == "BGEN";
            if (batched) in >> slot;
            else if (op == "GEN") slot = 0;
            if (slot < 0 || slot >= count || !(in >> limit >> text) || in >> extra || limit < 1 ||
                (batched ? slots[slot].active : running()))
                throw std::invalid_argument("expected GEN <max_new> <ids> or BGEN <free-slot> <max_new> <ids>");
            const auto prompt = token_ids(text);
            if (prompt.empty() || prompt.size() + (size_t)limit > (size_t)context ||
                std::any_of(prompt.begin(), prompt.end(), [&](int t) { return t < 0 || t >= decoder.vocabulary(); }))
                throw std::invalid_argument("request exceeds context or has invalid token counts");
            decoder.select_sequence(slot);
            decoder.reset();
            decoder.enable_mtp_capture(mtp && (!batched || batch_mtp));
            const auto start = Clock::now();
            auto cancelled = [&] { return bool(*request.cancel) || (batched && stopped[slot].load()); };
            decoder.set_cancel(cancelled);
            std::vector<float> logits;
            bool did_yield = false;
            for (size_t at = 0; at < prompt.size() && !cancelled(); at += width) {
                const size_t end = std::min(prompt.size(), at + width);
                logits = decoder.batch(std::vector<int>(prompt.begin() + at, prompt.begin() + end));
                std::cout << "PP " << end << ' ' << prompt.size() << '\n' << std::flush;
                if (batched && yielded[slot].exchange(false) && end < prompt.size()) {
                    std::cout << "YIELDED " << slot << ' ' << end << '\n'; did_yield = true; break;
                }
                // A long admission allows already admitted sequences a decode step per chunk.
                decoder.set_cancel({}); if (!batch_mtp) safe_pump(); decoder.set_cancel(cancelled);
            }
            auto prefilled = Clock::now();
            int generated = 0;
            bool cont = false;
            std::string reason = cancelled() ? "cancel" : did_yield ? "yield" : "stop";
            if (!cancelled() && !did_yield) {
                if (mtp && (!batched || batch_mtp)) { decoder.prepare_mtp(prompt); if (!batched) decoder.prepare_mtp_history(depth); }
                decoder.prepare_decode_cache(); decoder.freeze_cache(); decoder.prepare_cpu(0);
                prefilled = Clock::now();
                if (batched) {
                    const int token = std::max_element(logits.begin(), logits.end()) - logits.begin();
                    std::cout << "T " << token << '\n'; generated = delivered_control = 1;
                    cont = !is_stop(token) && limit > 1;
                    slots[slot] = {cont, token, 1, limit, Clock::now()};
                    reason = is_stop(token) ? "stop" : "length";
                } else if (mtp) {
                    auto result = generate_mtp(decoder, logits, limit, stops, depth, [&](int token) {
                        if (cancelled()) throw std::runtime_error("cancelled");
                        ++delivered_control;
                        std::cout << "T " << token << '\n' << std::flush;
                    });
                    generated = result.generated; reason = result.stopped ? "stop" : "length";
                } else {
                    reason = "length";
                    for (; generated < limit && !cancelled();) {
                        const int token = std::max_element(logits.begin(), logits.end()) - logits.begin();
                        ++generated; ++delivered_control; std::cout << "T " << token << '\n' << std::flush;
                        if (is_stop(token)) { reason = "stop"; break; }
                        if (generated < limit) logits = decoder.step(token);
                    }
                    if (cancelled()) reason = "cancel";
                }
            }
            std::cout << "DONE " << generated << ' ' << prompt.size() << ' ' << std::chrono::duration<double, std::milli>(prefilled - start).count()
                      << ' ' << ms(prefilled) << ' ' << reason << " 0 0 0\n";
            if (batched) std::cout << "BADM " << slot << ' ' << int(cont) << '\n';
            std::cout << std::flush;
        } catch (const std::exception &e) {
            decoder.set_cancel({});
            if (*request.cancel || (batched && slot >= 0 && slot < count && stopped[slot])) {
                std::cout << "DONE " << delivered_control << " 0 0 0 cancel 0 0 0\n";
                if (batched) std::cout << "BADM " << slot << " 0\n";
                std::cout << std::flush;
            } else {
                std::cout << "ERR " << e.what() << '\n' << std::flush;
            }
        }
        decoder.set_cancel({});
        { std::lock_guard lock(mutex); active.reset(); }
    }
    decoder.set_cancel({}); decoder.report_gpu(); decoder.report_cache();
    return 0;
}
