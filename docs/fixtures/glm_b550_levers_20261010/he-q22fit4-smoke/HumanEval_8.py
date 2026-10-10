from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    total = 0
    product = 1
    for n in numbers:
        total += n
        product *= n
    return (total, product)
