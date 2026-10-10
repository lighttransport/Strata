from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(paren_string):
        if c == ' ':
            continue
        if c == '(':
            if depth == 0:
                start = i
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                result.append(paren_string[start:i+1].replace(' ', ''))
    return result
