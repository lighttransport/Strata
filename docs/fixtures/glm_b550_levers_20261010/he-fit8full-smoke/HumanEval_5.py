from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    result = []
    for i, n in enumerate(numbers):
        if i > 0:
        result.append(delimeter)
        result.append(n)
    return result
