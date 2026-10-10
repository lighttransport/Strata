def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    from itertools import combinations
    pair = min(combinations(numbers, 2), key=lambda p: abs(p[0] - p[1]))
    return tuple(sorted(pair)) if pair[0] != pair[1] else (pair[0], pair[1])

Wait — simpler: return (min(pair), max(pair)).

def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    from itertools import combinations
    pair = min(combinations(numbers, 2), key=lambda p: abs(p[0] - p[1]))
    return (min(pair), max(pair))