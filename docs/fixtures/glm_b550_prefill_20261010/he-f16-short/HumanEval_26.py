from typing import List

def remove_duplicates(numbers: List[int]) -> List[int]:
    return [x for x in numbers if numbers.count(x) == 1]
