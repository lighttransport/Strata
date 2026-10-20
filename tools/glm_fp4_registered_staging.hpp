#pragma once
#include "ucomm.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::experiment {
// Populate before service, then give each context one owner. Clear only after
// its requests finish and before finalizing any associated ucomm context.
class RegisteredStaging {
public:
    class Buffers {
        using Registration = std::unique_ptr<ucomm_mr_t, decltype(&ucomm_mr_dereg)>;
        ucomm_t* context_;
        std::vector<uint8_t> tx_, rx_;
        // Registrations are destroyed before their backing vectors.
        Registration tx_mr_{nullptr, ucomm_mr_dereg}, rx_mr_{nullptr, ucomm_mr_dereg};
        static Registration register_buffer(ucomm_t* context, std::vector<uint8_t>& bytes) {
            ucomm_mr_t* raw = nullptr;
            const auto status = ucomm_mr_reg(context, bytes.data(), bytes.size(), &raw);
            if (status != UCOMM_OK) throw std::runtime_error(ucomm_strerror(status));
            return Registration(raw, ucomm_mr_dereg);
        }
    public:
        Buffers(ucomm_t* context, size_t capacity) : context_(context), tx_(capacity), rx_(capacity) {
            if (!context || !capacity) throw std::invalid_argument("invalid staging context or capacity");
            tx_mr_ = register_buffer(context, tx_);
            rx_mr_ = register_buffer(context, rx_);
        }
        Buffers(const Buffers&) = delete;
        Buffers& operator=(const Buffers&) = delete;
        ucomm_t* context() const { return context_; }
        uint8_t* tx(size_t bytes) {
            if (bytes > tx_.size()) throw std::length_error("staging send extent");
            return tx_.data();
        }
        uint8_t* rx(size_t bytes) {
            if (bytes > rx_.size()) throw std::length_error("staging receive extent");
            return rx_.data();
        }
    };
private:
    std::vector<std::unique_ptr<Buffers>> entries_;
public:
    Buffers* find(ucomm_t* context) const {
        for (const auto& entry : entries_) if (entry->context() == context) return entry.get();
        return nullptr;
    }
    void add(ucomm_t* context, size_t capacity) {
        if (find(context)) throw std::invalid_argument("duplicate staging context");
        entries_.push_back(std::make_unique<Buffers>(context, capacity));
    }
    size_t size() const { return entries_.size(); }
    void clear() { entries_.clear(); }
};
}
