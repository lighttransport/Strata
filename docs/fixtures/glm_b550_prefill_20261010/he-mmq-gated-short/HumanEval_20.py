from typing import List, Tuple

def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    s = sorted(numbers)
    return min(zip(s, s[1:]), key=lambda p: p[1] - p[0])
