from typing import List
from collections import Counter

def remove_duplicates(numbers: List[int]) -> List[int]:
    counts = Counter(numbers)
    return [x for x in numbers if counts[x] == 1]
