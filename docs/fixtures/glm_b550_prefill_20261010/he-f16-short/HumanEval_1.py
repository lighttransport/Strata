from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    s = paren_string.replace(' ', '')
    result, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c == '(':
            if depth == 0:
                start = i
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                result.append(s[start:i+1])
    return result
