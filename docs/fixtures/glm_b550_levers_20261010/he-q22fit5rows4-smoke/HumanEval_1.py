from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    depth = 0
    current = []
    for ch in paren_string.replace(' ', ''):
        current.append(ch)
        if ch == '(':
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                result.append(''.join(current))
                current = []
    return result
