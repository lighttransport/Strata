#include "strata/ds41/stream_config.hpp"
#include <cassert>

static bool accepts() {
    try { strata::ds41::validate_owned_stream_settings(); return true; }
    catch (const std::invalid_argument&) { return false; }
}
int main() {
    const char* names[] = {"STRATA_DS41_OWN_STREAM", "STRATA_DS41_REMOTE_OWN", "STRATA_DS41_POPULATE",
                          "STRATA_DS41_PIN_GB", "STRATA_DS41_PIN_EXPERTS", "STRATA_DS41_RAM_COPY"};
    for (const char* name : names) unsetenv(name);
    assert(accepts());  // defaults without streaming retain their existing behavior
    setenv(names[0], "192.0.2.1:29574", 1);
    assert(!accepts());
    setenv(names[1], "192.0.2.1:29571", 1);
    setenv(names[2], "0", 1);
    assert(!accepts());  // the default pin budget is unsafe even with population off
    setenv(names[3], "0", 1);
    assert(accepts());
    for (int i = 1; i < 6; ++i) {
        setenv(names[i], i == 1 ? "" : "1", 1);
        assert(!accepts());
        setenv(names[i], i == 1 ? "192.0.2.1:29571" : "0", 1);
    }
    unsetenv(names[2]);
    assert(!accepts());  // the default population setting is unsafe
    setenv(names[0], "", 1);
    assert(accepts());
}
