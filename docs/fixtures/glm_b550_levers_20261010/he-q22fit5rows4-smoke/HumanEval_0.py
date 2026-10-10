from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    sorted_nums = sorted(numbers)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] < threshold:
            return True
    return False
