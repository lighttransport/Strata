from collections import Counter

def remove_duplicates(numbers: List[int]) -> List[int]:
    counts = Counter(numbers)
    return [n for n in numbers if counts[n] == 1]