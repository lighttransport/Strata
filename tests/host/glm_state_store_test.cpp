// tests/host/glm_state_store_test.cpp - the GLM serve state store (src/program/glm_state_store.hpp): image round trip,
// RAM and disk tiers, longest-prefix lookup, LRU caps, a restart reading the disk tier, damaged files are misses.
#include "../../src/program/glm_state_store.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <unistd.h>

using strata::glm::GlmStateImage;
using strata::glm::GlmStateStore;

static int fails = 0;
#define EXPECT(cond) do { if (!(cond)) { std::printf("FAIL line %d: %s\n", __LINE__, #cond); ++fails; } } while (0)

static std::shared_ptr<GlmStateImage> image(const std::vector<int> &prompt, int position, uint8_t fill, size_t row_bytes = 4096) {
    auto im = std::make_shared<GlmStateImage>();
    im->position = position; im->mtp_position = position;
    im->tokens.assign(prompt.begin(), prompt.begin() + position);
    im->fixed.assign(1000, fill);
    im->rows = {std::vector<uint8_t>(row_bytes, uint8_t(fill + 1)), std::vector<uint8_t>(row_bytes / 2, uint8_t(fill + 2))};
    im->last_hidden.assign(16, float(fill));
    im->routes = {1, 2, 3};
    return im;
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("glm_state_store_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::vector<int> prompt(300);
    for (int i = 0; i < 300; ++i) prompt[i] = i * 7 % 1000;

    // serialize / deserialize
    auto a = image(prompt, 128, 5);
    const auto bytes = strata::glm::serialize_image(*a);
    GlmStateImage back;
    EXPECT(strata::glm::deserialize_image(bytes.data(), bytes.size(), back));
    EXPECT(back.position == 128 && back.tokens == a->tokens && back.fixed == a->fixed && back.rows == a->rows &&
           back.last_hidden == a->last_hidden && back.routes == a->routes);
    EXPECT(!strata::glm::deserialize_image(bytes.data(), bytes.size() - 1, back));

    {
        GlmStateStore::Options o;
        o.ram_bytes = 8 * a->bytes();                  // room for every image of this block
        o.disk_dir = dir.string(); o.disk_bytes = uint64_t(1) << 30;
        o.identity_model = 11; o.identity_config = 22;
        GlmStateStore store(o);
        store.put(a);
        store.put(image(prompt, 256, 9));
        std::string tier;
        // the longest stored prefix below the prompt's length wins
        auto hit = store.find(prompt, {64, 128, 256}, &tier);
        EXPECT(hit && hit->position == 256 && tier == "ram");
        // a prompt that diverges at 200: only the 128 prefix matches
        auto other = prompt; other[200] = 999999;
        hit = store.find(other, {128, 256}, &tier);
        EXPECT(hit && hit->position == 128);
        // never the whole prompt: one token must be read for the logits
        std::vector<int> exact(prompt.begin(), prompt.begin() + 256);
        hit = store.find(exact, {256, 128});
        EXPECT(hit && hit->position == 128);
        store.put(image(prompt, 64, 3));
        store.flush();
        EXPECT(store.stats().written == 3 && store.stats().disk_entries == 3);
    }
    {
        // a restart: the disk tier is indexed again; another identity sees nothing
        GlmStateStore::Options o;
        o.disk_dir = dir.string(); o.disk_bytes = uint64_t(1) << 30;
        o.identity_model = 11; o.identity_config = 22;
        GlmStateStore store(o);
        std::string tier;
        auto hit = store.find(prompt, {256}, &tier);
        EXPECT(hit && hit->position == 256 && tier == "disk" && hit->fixed == std::vector<uint8_t>(1000, 9));
        o.identity_config = 23;
        GlmStateStore other(o);
        EXPECT(other.find(prompt, {256, 128, 64}) == nullptr);
        // a damaged file is a miss and is removed
        const auto path = store.file_for(GlmStateStore::prefix_hash(prompt.data(), 128), 128);
        {
            std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(200); f.put('\x7f');
        }
        EXPECT(store.find(prompt, {128}) == nullptr);
        EXPECT(!std::filesystem::exists(path));
        // the disk cap evicts the least recently used file
        GlmStateStore::Options small = o;
        small.identity_config = 22;
        small.disk_bytes = std::filesystem::file_size(store.file_for(GlmStateStore::prefix_hash(prompt.data(), 256), 256)) + 10;
        GlmStateStore capped(small);
        capped.put(image(prompt, 32, 1));
        capped.flush();
        EXPECT(capped.stats().disk_entries == 1);
    }
    {
        // RAM cap: with room for two images, a third evicts the least recently used
        GlmStateStore::Options o;
        o.ram_bytes = image(prompt, 256, 0)->bytes() * 2;
        GlmStateStore store(o);
        store.put(image(prompt, 64, 1));
        store.put(image(prompt, 128, 2));
        EXPECT(store.find(prompt, {64}) != nullptr);                 // 64 is now the most recent
        store.put(image(prompt, 256, 3));
        EXPECT(store.stats().ram_entries == 2);
        EXPECT(store.find(prompt, {128}) == nullptr && store.find(prompt, {64}) != nullptr);
    }
    // explicit session files
    {
        std::string error;
        const auto path = (dir / "session.glmstate").string();
        EXPECT(GlmStateStore::write_file(path, *a, 1, 2, error));
        GlmStateImage read;
        EXPECT(GlmStateStore::read_file(path, 1, 2, read, error) && read.tokens == a->tokens);
        EXPECT(!GlmStateStore::read_file(path, 1, 3, read, error));
    }
    std::filesystem::remove_all(dir);
    std::printf("glm_state_store_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
