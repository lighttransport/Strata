from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    s = sorted(numbers)
    return any(s[i+1] - s[i] < threshold for i in range(len(s) - 1))
