from typing import List

def rescale_to_unit(numbers: List[float]) -> List[float]:
    lo, hi = min(numbers), max(numbers)
    width = hi - lo
    return [(n - lo) / width for n in numbers]
