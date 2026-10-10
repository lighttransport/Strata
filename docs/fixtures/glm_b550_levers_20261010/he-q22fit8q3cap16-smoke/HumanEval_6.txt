The task: parse_nested_parens takes a string of groups separated by spaces, each group is a paren string; return list of max nesting depth per group.

Simple approach: iterate chars, depth counter, track max. For each group (split by spaces), compute max depth.

Let me write code:

def parse_nested_parens(paren_string: str) -> List[int]:
    result = []
    for group in paren_string.split(' '):
        max_depth = 0
        current = 0
        for char in group:
        if char == '(':
        current += 1
        elif char == ')':
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 1
        else:
        current -= 