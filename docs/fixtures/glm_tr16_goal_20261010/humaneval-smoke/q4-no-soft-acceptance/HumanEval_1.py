from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    depth = 0
    group = ''
    for ch in paren_string:
        if ch == '(':
            depth += 1
            group += ch
        elif ch == ')':
            depth -= 1
            group += ch
            if depth == 0:
                result.append(group)
                group = ''
    return result
