from typing import List

def factorize(n: int) -> List[int]:
    factors: List[int] = []
    d = 2
    while d <= n:
        if n % d == 0:
        if n % d == 0:
            factors.append(d)
            n //= d
        else:
        else:
            d += 1
    return factors
