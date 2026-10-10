The function iterates through the list, tracking the maximum seen so far and appending it to the result.

```from typing import List

def rolling_max(numbers: List[int]) -> List[int]:
    result = []
    current_max = None
    for n in numbers:
        if current_max is None or n > current_max:
            current_max = n
        result.append(current_max)
    return result
```

- Handles empty list correctly (returns `[]`)
- Uses running maximum, not global maximum
- Appends maximum at each position, matching expected output