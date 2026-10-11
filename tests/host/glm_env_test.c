// tests/host/glm_env_test.c - the knob table (no model, no GPU).
//
// The table is the engine's tuning surface, so this test is about what happens when a value is WRONG as much
// as when it is right: the lambdas the table replaced each hand-wrote their own range check, and the failure
// this guards against is a knob whose table entry has drifted from what it used to accept.
#include "strata/host/glm_env.h"

/* setenv / unsetenv are POSIX, not C11; asking for them explicitly keeps this file building under -std=c11
   -pedantic as well as under CMake's default gnu11. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c)                                              \
    do {                                                      \
        if (!(c)) {                                           \
            (void)printf("FAIL line %d: %s\n", __LINE__, #c); \
            ++fails;                                          \
        }                                                     \
    } while (0)

#if defined(_WIN32)
static void unset(const char *k) { _putenv_s(k, ""); }
static void put(const char *k, const char *v) { _putenv_s(k, v); }
#else
static void unset(const char *k) { unsetenv(k); }
static void put(const char *k, const char *v) { setenv(k, v, 1); }
#endif

static const strata_env_knob kKnobs[] = {
    /* name, kind, lo, hi, default */
    {"TEST_ROUTES", STRATA_ENV_INT, 0, 8, 0},
    {"TEST_SHARE", STRATA_ENV_FLOAT, 0.0, 1.0, 1.0},
    {"TEST_FLAG", STRATA_ENV_FLAG, 0, 1, 0},
    {"TEST_LOOSE", STRATA_ENV_FLAG_LOOSE, 0, 1, 0},
    {"TEST_DIR", STRATA_ENV_TEXT, 0, 0, 0},
    {"TEST_HI_LO", STRATA_ENV_INT, 4, 4, 4},
};

#define COUNT_OF(x) (sizeof(x) / sizeof((x)[0]))

static char message[STRATA_ENV_TEXT_MAX];

static int load(void) {
    return strata_env_init(kKnobs, COUNT_OF(kKnobs), message, sizeof message);
}

int main(void) {
    int id;
    char big[STRATA_ENV_TEXT_MAX + 16];

    /* Defaults with nothing set. */
    unset("TEST_ROUTES");
    unset("TEST_SHARE");
    unset("TEST_FLAG");
    unset("TEST_LOOSE");
    unset("TEST_DIR");
    unset("TEST_HI_LO");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_count() == COUNT_OF(kKnobs));
    CHECK(strata_env_get_int(strata_env_id("TEST_ROUTES")) == 0);
    CHECK(strata_env_get(strata_env_id("TEST_SHARE")) == 1.0);
    CHECK(strata_env_get_int(strata_env_id("TEST_FLAG")) == 0);
    CHECK(strata_env_get_int(strata_env_id("TEST_LOOSE")) == 0);
    CHECK(strata_env_get_int(strata_env_id("TEST_HI_LO")) == 4);
    CHECK(strata_env_text(strata_env_id("TEST_DIR")) != NULL);
    CHECK(strata_env_text(strata_env_id("TEST_DIR"))[0] == '\0');
    CHECK(strata_env_last_error() == STRATA_ENV_OK);

    /* A table registered with nothing in it: a getter reports that, rather than inventing a number. */
    CHECK(strata_env_init(NULL, 0, message, sizeof message) == STRATA_ENV_OK);
    CHECK(strata_env_count() == 0);
    CHECK(strata_env_get_int(0) == 0);
    CHECK(strata_env_last_error() == STRATA_ENV_ERR_UNINIT);
    CHECK(strata_env_text(0) == NULL);
    CHECK(strata_env_last_error() == STRATA_ENV_ERR_UNINIT);
    CHECK(load() == STRATA_ENV_OK);

    /* In range, and the edges included at both ends. */
    put("TEST_ROUTES", "8");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get_int(strata_env_id("TEST_ROUTES")) == 8);
    put("TEST_ROUTES", "0");
    CHECK(load() == STRATA_ENV_OK);
    put("TEST_SHARE", "0.25");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get(strata_env_id("TEST_SHARE")) == 0.25);
    put("TEST_SHARE", "0");
    CHECK(load() == STRATA_ENV_OK);
    put("TEST_SHARE", "1");
    CHECK(load() == STRATA_ENV_OK);

    /* Out of range: the message names the variable and the range, because the value is what was typed. */
    put("TEST_ROUTES", "9");
    CHECK(load() == STRATA_ENV_ERR_RANGE);
    CHECK(strstr(message, "TEST_ROUTES") != NULL);
    CHECK(strstr(message, "9") != NULL);
    put("TEST_ROUTES", "-1");
    CHECK(load() == STRATA_ENV_ERR_RANGE);
    put("TEST_ROUTES", "3.5");
    CHECK(load() == STRATA_ENV_ERR_FORMAT);
    put("TEST_ROUTES", "8 ");
    CHECK(load() == STRATA_ENV_ERR_FORMAT); /* trailing junk is not an 8 */
    put("TEST_ROUTES", "");
    CHECK(load() == STRATA_ENV_ERR_FORMAT);
    put("TEST_ROUTES", "0x8");
    CHECK(load() == STRATA_ENV_ERR_FORMAT);
    unset("TEST_ROUTES");
    unset("TEST_SHARE");
    CHECK(load() == STRATA_ENV_OK);

    /* A single allowed value reads as one, not as a range. */
    put("TEST_HI_LO", "3");
    CHECK(load() == STRATA_ENV_ERR_RANGE);
    CHECK(strstr(message, "4") != NULL);
    CHECK(strstr(message, "..") == NULL);
    put("TEST_HI_LO", "4");
    CHECK(load() == STRATA_ENV_OK);

    /* Strict 0/1 rejects anything else; the loose kind is false for everything that is not exactly "1". */
    put("TEST_FLAG", "2");
    CHECK(load() == STRATA_ENV_ERR_FORMAT);
    CHECK(strstr(message, "0 nor 1") != NULL);
    put("TEST_FLAG", "true");
    CHECK(load() == STRATA_ENV_ERR_FORMAT);
    put("TEST_FLAG", "1");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get_int(strata_env_id("TEST_FLAG")) == 1);
    put("TEST_LOOSE", "TRACE");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get_int(strata_env_id("TEST_LOOSE")) == 0);
    put("TEST_LOOSE", "1");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get_int(strata_env_id("TEST_LOOSE")) == 1);

    /* Text is kept verbatim, and a value that does not fit is an error rather than a truncation. */
    put("TEST_DIR", "/var/tmp/calibration");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_text(strata_env_id("TEST_DIR")) != NULL);
    CHECK(strcmp(strata_env_text(strata_env_id("TEST_DIR")), "/var/tmp/calibration") == 0);
    memset(big, 'x', sizeof big - 1);
    big[sizeof big - 1] = '\0';
    put("TEST_DIR", big);
    CHECK(load() == STRATA_ENV_ERR_TEXT);
    unset("TEST_DIR");
    CHECK(load() == STRATA_ENV_OK);

    /* Lookup by name, and the errors that are not about values at all. */
    CHECK(strata_env_id("TEST_ROUTES") == 0);
    CHECK(strata_env_id("TEST_NO_SUCH") == -1);
    CHECK(strata_env_id(NULL) == -1);
    unset("TEST_ROUTES");
    unset("TEST_SHARE");
    unset("TEST_FLAG");
    unset("TEST_LOOSE");
    unset("TEST_DIR");
    CHECK(load() == STRATA_ENV_OK);
    put("TEST_FLAG", "1");
    {
        static const strata_env_knob dup[2] = {{"TEST_A", STRATA_ENV_INT, 0, 1, 0},
                                              {"TEST_A", STRATA_ENV_INT, 0, 1, 0}};
        CHECK(strata_env_init(dup, 2, message, sizeof message) == STRATA_ENV_ERR_DUP);
        CHECK(strata_env_init(kKnobs, 100000, message, sizeof message) == STRATA_ENV_ERR_ARGS);
    }
    CHECK(load() == STRATA_ENV_OK); /* the failed inits left the working table alone */

    /* An id outside the table, and a value fetched with the wrong getter, are errors - never a number. */
    id = strata_env_id("TEST_DIR");
    CHECK(id >= 0);
    put("TEST_DIR", "/tmp");
    CHECK(load() == STRATA_ENV_OK);
    CHECK(strata_env_get(id) == 0.0);
    CHECK(strata_env_get_int(id) == 1); /* text: 1 when set */
    CHECK(strata_env_last_error() == STRATA_ENV_OK);
    CHECK(strata_env_get_int(999) == 0);
    CHECK(strata_env_last_error() == STRATA_ENV_ERR_ID);
    CHECK(strata_env_get(-1) == 0.0);
    CHECK(strata_env_last_error() == STRATA_ENV_ERR_ID);
    unset("TEST_FLAG");
    unset("TEST_DIR");

    /* The engine's existing switches.  Each rule is kept exactly as the call sites had it, and the cases that
       differ between rules are the ones that matter: a cleanup that unified them would flip set knobs. */
    unset("TEST_TRUTHY");
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_nonempty("TEST_TRUTHY") == 0);
    CHECK(strata_env_is_one("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 1) == 1);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 0) == 0);
    put("TEST_TRUTHY", "1");
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 1);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 1);
    CHECK(strata_env_truthy_nonempty("TEST_TRUTHY") == 1);
    CHECK(strata_env_is_one("TEST_TRUTHY") == 1);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 0) == 1);
    {   /* "2": the two rules disagree, and both are what their knobs have always done */
        put("TEST_TRUTHY", "2");
        CHECK(strata_env_truthy_number("TEST_TRUTHY") == 1);
        CHECK(strata_env_truthy_one("TEST_TRUTHY") == 0);
    }
    put("TEST_TRUTHY", "0");
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_nonempty("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 1) == 0);
    put("TEST_TRUTHY", "off");   /* atoi reads 0; the prefix rules read "on" */
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 1) == 1);
    CHECK(strata_env_is_one("TEST_TRUTHY") == 0);
    put("TEST_TRUTHY", "1x");   /* atoi reads 1; the "starts with 1" rule reads on too */
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 1);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 1);
    put("TEST_TRUTHY", "12");
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 1);
    CHECK(strata_env_truthy_one("TEST_TRUTHY") == 1);   /* "12" starts with '1' */
    put("TEST_TRUTHY", "");
    CHECK(strata_env_truthy_number("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_nonempty("TEST_TRUTHY") == 0);
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 1) == 1);   /* "" does not start with '0' */
    put("TEST_TRUTHY", "01");
    CHECK(strata_env_truthy_default("TEST_TRUTHY", 1) == 0);   /* starts with '0' */
    unset("TEST_TRUTHY");
    CHECK(strata_env_value_double("TEST_TRUTHY", 24.0) == 24.0);
    put("TEST_TRUTHY", "3.5");
    CHECK(strata_env_value_double("TEST_TRUTHY", 24.0) == 3.5);
    put("TEST_TRUTHY", "junk");
    CHECK(strata_env_value_double("TEST_TRUTHY", 24.0) == 0.0);   /* atof, not a rejected value */
    CHECK(strata_env_value_ll("TEST_TRUTHY", 2) == 0);
    put("TEST_TRUTHY", "7");
    CHECK(strata_env_value_ll("TEST_TRUTHY", 2) == 7);
    unset("TEST_TRUTHY");

    /* The two text rules: "the only off value is 0", and "unless the user asked for this exact word". */
    unset("TEST_TEXT");
    CHECK(strata_env_text_on("TEST_TEXT") == 0);
    CHECK(strata_env_equals("TEST_TEXT", "top") == 0);   /* unset equals nothing */
    put("TEST_TEXT", "");
    CHECK(strata_env_text_on("TEST_TEXT") == 0);
    put("TEST_TEXT", "0");
    CHECK(strata_env_text_on("TEST_TEXT") == 0);
    put("TEST_TEXT", "1");
    CHECK(strata_env_text_on("TEST_TEXT") == 1);
    put("TEST_TEXT", "off");   /* text_on's "on": only "0" turns it off */
    CHECK(strata_env_text_on("TEST_TEXT") == 1);
    put("TEST_TEXT", "top");
    CHECK(strata_env_equals("TEST_TEXT", "top"));
    CHECK(strata_env_equals("TEST_TEXT", "TOP") == 0);
    CHECK(strata_env_text_on("TEST_TEXT") == 1);
    unset("TEST_TEXT");

    /* A set "0" is a 0, not the fallback: that is what the call site's atoi did. */
    CHECK(strata_env_int_or("TEST_INT", 4) == 4);
    put("TEST_INT", "0");
    CHECK(strata_env_int_or("TEST_INT", 4) == 0);
    put("TEST_INT", "");
    CHECK(strata_env_int_or("TEST_INT", 4) == 4);   /* empty, not set: the fallback */
    put("TEST_INT", "junk");
    CHECK(strata_env_int_or("TEST_INT", 4) == 0);   /* atoi, not a rejected value */
    put("TEST_INT", "9");
    CHECK(strata_env_int_or("TEST_INT", 4) == 9);
    unset("TEST_INT");

    if (fails != 0) return 1;
    (void)printf("env OK %d knobs\n", (int)strata_env_count());
    return 0;
}
