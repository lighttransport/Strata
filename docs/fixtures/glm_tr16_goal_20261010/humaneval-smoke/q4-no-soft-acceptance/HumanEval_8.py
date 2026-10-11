from typing import List, Tuple

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    total = sum(numbers)
    product = 1
    for n in numbers:
        product *= n
    return (total, product)
