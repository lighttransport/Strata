from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    result = []
    depth = 0
    max_depth = 0
    for char in paren_string:
    if char == '(':
        depth += 1
        max_depth = max(depth, max_depth)
    elif char == ')':
        depth -= 1
        max_depth = max(depth, max_depth)
        if depth < 0:
            return False
        elif depth == 0:
            if max_depth > 0:
                result.append(max_depth)
    else:
        depth += 1
        max_depth = max(depth, max_depth)
    return result
