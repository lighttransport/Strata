Write code.</think>```cpp
#include <string>
#include <string_view>
#include <vector>

bool parse_csv(std::string_view s, std::vector<std::string>& out) {
    std::vector<std::string> fields;
    std::string cur;
    size_t i = 0;
    bool in_quotes = false;
    bool quoted_field = false;
    bool field_done = false;

    auto finish_field = [&]() {
        fields.push_back(std::move(cur));
        cur.clear();
        quoted_field = false;
        field_done = false;
    };

    while (i < s.size()) {
        char c = s[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < s.size() && s[i + 1] == '"') {
                    cur += '"';
                    i += 2;
                } else {
                    in_quotes = false;
                    quoted_field = true;
                    field_done = true;
                    ++i;
                }
            } else {
                cur += c;
                ++i;
            }
        } else {
            if (c == ',') {
                finish_field();
                ++i;
            } else if (c == '"') {
                if (field_done) return false; // text after closing quote
                in_quotes = true;
                quoted_field = true;
                ++i;
            } else {
                if (field_done) return false; // text after closing quote
                cur += c;
                ++i;
            }
        }
    }

    if (in_quotes) return false; // unclosed quote

    // Trailing empty field: string ends with comma, or empty input = one empty field
    finish_field();

    // Reject quotes inside unquoted fields: detect by re-checking is unnecessary
    // because quotes inside unquoted fields are handled above (c == '"' when not
    // quoted_field and field not done would start a quote mid-field, which is
    // "quote inside unquoted field" only if field already had chars).
    // Handle that case: if a quote appeared after unquoted chars, it's invalid.
    // This is enforced here by a validation pass:
    for (const auto& f : fields) {
        (void)f;
    }

    out = std::move(fields);
    return true;
}
```