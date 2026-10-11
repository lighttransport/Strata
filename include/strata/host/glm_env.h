// include/strata/host/glm_env.h - every tuning knob in one table.
//
// The engine's knobs arrive through the environment: 133 of them in the decoder alone, and each one has
// its own `static const T x = [] { const char *v = std::getenv(...); ... }()` block.  That shape is honest
// about a default and it hides everything else: the range is a hand-written `if` inside a lambda, the
// message is spelled three different ways, and nothing can enumerate the knobs - so adding one means
// copying five lines from a nearby one and hoping the ranges match.
//
// A table does enumerate them.  One line per knob names it, says what kind of value it holds, what range
// is accepted, and what the default is; the parsing, range checking and message are then written once and
// tested once.
//
// C11, portable subset (see glm_names.h for why).  Not thread safe by design: call strata_env_init from
// main() once, before any worker thread starts, because C has no function-local static initialization to
// lean on the way the C++ lambdas it replaces do.
//
// WHAT THE TABLE IS FOR: a knob whose range is part of its contract - a count, a fraction, a timeout - where
// the range and the message are then written once and tested once.  No knob in the engine is in it yet, and
// the engine's switches are not a rename away from it: see the switch section below, which is what they did
// become this round.
#ifndef STRATA_HOST_GLM_ENV_H_
#define STRATA_HOST_GLM_ENV_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Longest environment value kept for a STRATA_ENV_TEXT knob: a calibration directory path.  A longer value
   is reported as an error rather than truncated, for the same reason as a truncated tensor name. */
#define STRATA_ENV_TEXT_MAX 512

/* Knobs the table can hold. */
typedef enum {
    /* An integer.  Compared against lo/hi inclusive, and a value with trailing junk ("8 ", "eight") is an
       error rather than a prefix parse - the lambda this replaces used strtol and then checked the tail. */
    STRATA_ENV_INT = 0,
    /* A float, same rules as INT. */
    STRATA_ENV_FLOAT,
    /* Strict 0/1.  Most callers want this: a knob that only means "on" must not treat "2", "yes" or an
       unset variable's leftover "0 " as true. */
    STRATA_ENV_FLAG,
    /* Loose 0/1: anything that is not exactly "1" is false, unset included.  Kept because several existing
       callers are written that way, and a migration must not quietly turn "TRACE" into an error. */
    STRATA_ENV_FLAG_LOOSE,
    /* A string kept verbatim, lo/hi unused. */
    STRATA_ENV_TEXT
} strata_env_kind;

/* One knob.  `name` is the variable WITHOUT a leading space; `def` is used when it is unset. */
typedef struct {
    const char *name;
    strata_env_kind kind;
    double lo;
    double hi;
    double def;
} strata_env_knob;

enum {
    STRATA_ENV_OK = 0,
    STRATA_ENV_ERR_ARGS,    /* null table, or count above the table capacity */
    STRATA_ENV_ERR_DUP,     /* two knobs named the same thing */
    STRATA_ENV_ERR_RANGE,   /* set, and outside lo..hi */
    STRATA_ENV_ERR_FORMAT,  /* set, and not a number / not 0-1 / not a value for its kind */
    STRATA_ENV_ERR_TEXT,    /* STRATA_ENV_TEXT value longer than STRATA_ENV_TEXT_MAX */
    STRATA_ENV_ERR_UNINIT,  /* a getter was called before strata_env_init */
    STRATA_ENV_ERR_ID       /* a getter was called with an id the table does not hold */
};

/* Registers `knobs` (count entries) and reads the environment once.  Replaces any previous table, so a
   test can register a small one and then the real one.  Returns STRATA_ENV_OK, or the first failure with a
   message in `err` naming the variable, the offending value and the range that was expected. */
int strata_env_init(const strata_env_knob *knobs, size_t count, char *err, size_t cap);

/* Number of knobs the table holds (0 before init). */
size_t strata_env_count(void);

/* Index of `name` in the table, or -1.  Look it up once and pass the id to the getters. */
int strata_env_id(const char *name);

/* The knob's value, as an int (INT/FLAG), a double (INT/FLOAT), or the string (TEXT).  A getter on the
   wrong kind, on an unknown id, or before init returns the default and sets the process's last error,
   retrievable with strata_env_last_error() - never a value that looks valid. */
int strata_env_get_int(int id);
double strata_env_get(int id);
const char *strata_env_text(int id);

/* The last error any of the above produced: STRATA_ENV_OK or one of the enum. */
int strata_env_last_error(void);

/* ---- The engine's existing switches, read the way they were already read ------------------------------
 *
 * These are stateless and need no init, because the engine reads them from library code whose binaries each
 * have their own main() and a knob must not depend on which one ran.  They exist to give the historical
 * truthiness rules ONE spelling and ONE test each - not to change what a value means.
 *
 * There is no single rule, and that is the point.  Nine are in use and the differences are load-bearing:
 * `STRATA_LFUSE=2` is OFF (its rule is "starts with 1") while `STRATA_FUSE_HEAD_GR=2` is ON (its rule is "a
 * non-zero number"), and `STRATA_SPEC_PROB=off` is ON while `STRATA_SPEC_PROB_DT=off` reads as 1 - so a
 * migration that picks one "obvious" reading silently flips knobs somebody already set.  The `when_unset`
 * arguments are parameters because several of them default ON on CUDA and OFF on HIP.
 */

/* Set at all, empty or not.  Some knobs only care about presence. */
int strata_env_is_set(const char *name);

/* Set to a value `std::atoi` reads as non-zero ("1", "3", "-1"; "0", "1x"->1, "off"->0). */
int strata_env_truthy_number(const char *name);

/* `when_unset` unless `std::atoi` reads the value as non-zero.  "off" is OFF here and ON for
   strata_env_truthy_default, which is the pair of knobs the engine has both of. */
int strata_env_truthy_number_default(const char *name, int when_unset);

/* Set to a value starting with the character '1'.  "2" is OFF here, which is the whole difference from the
   rule above. */
int strata_env_truthy_one(const char *name);

/* `when_unset` unless the value starts with '0'.  Anything else - "1", "off", "" - is ON. */
int strata_env_truthy_default(const char *name, int when_unset);

/* Set, non-empty, and not starting with '0'.  An empty value is OFF, where the rule above would have it ON. */
int strata_env_truthy_nonempty(const char *name);

/* Set to exactly "1". */
int strata_env_is_one(const char *name);

/* `when_unset` unless the value is set, when `std::atof` reads it.  A non-numeric value reads as 0. */
double strata_env_value_double(const char *name, double when_unset);

/* `when_unset` unless the value is set, when `std::atoll` reads it. */
long long strata_env_value_ll(const char *name, long long when_unset);

/* Set, and exactly `text`.  Unset is NOT equal to anything, which is what "unless the user asked for `text`"
   means at its call sites. */
int strata_env_equals(const char *name, const char *text);

/* Set, non-empty, and not exactly "0".  The shape a couple of knobs use where "0" is the only off value. */
int strata_env_text_on(const char *name);

/* `std::atoi` unless the value is unset or empty, then `fallback`.  Note this is NOT "0 falls back": a set "0"
   is a 0, because that is what the call site's `atoi` did. */
int strata_env_int_or(const char *name, int fallback);

#ifdef __cplusplus
}
#endif

#endif // STRATA_HOST_GLM_ENV_H_
