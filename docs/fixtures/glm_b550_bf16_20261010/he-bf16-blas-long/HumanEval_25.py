def factorize(n: int) -> List[int]:
    factors = []
    d = 2
    while d <= n:
        if n % d == 0:
            factors.append(d)
            n //= d
        else:
            d += 1
    return factors

from typing import List

def factorize(n: int) -> List[int]:
    factors = []
    d = 2
    while d <= n:
        if n % d == 0:
            factors.append(d)
            n //= d
        else:
            d += 1
    return factors

# --- Tests (from typing import List) ---
assert factorize(8) == [2, 2, 2]
assert factorize(25) == [5, 5]
assert factorize(70) == [2, 5, 7]
assert factorize(1) == []
assert factorize(2) == [2]
assert factorize(3) == [3]
assert factorize(4) == [2, 2]
assert factorize(6) == [2, 3]
assert factorize(9) == [3, 3]
assert factorize(10) == [2, 5]
assert factorize(12) == [2, 2, 3]
assert factorize(15) == [3, 5]
assert factorize(16) == [2, 2, 2, 2]
assert factorize(18) == [2, 3, 3]
assert factorize(20) == [2, 2, 5]
assert factorize(27) == [3, 3, 3]
assert factorize(28) == [2, 2, 7]
assert factorize(30) == [2, 3, 5]
assert factorize(32) == [2, 2, 2, 2, 2]
assert factorize(36) == [2, 2, 3, 3]
assert factorize(40) == [2, 2, 2, 5]
assert factorize(42) == [2, 3, 7]
assert factorize(45) == [3, 3, 5]
assert factorize(48) == [2, 2, 2, 3]
assert factorize(49) == [7, 7]
assert factorize(50) == [2, 5, 5]
assert factorize(60) == [2, 2, 3, 5]
assert factorize(64) == [2, 2, 2, 2, 2, 2]
assert factorize(72) == [2, 2, 2, 3, 3]
assert factorize(75) == [3, 5, 5]
assert factorize(80) == [2, 2, 2, 2, 5]
assert factorize(81) == [3, 3, 3, 3]
assert factorize(90) == [2, 3, 3, 5]
assert factorize(96) == [2, 2, 2, 2, 2, 3]
assert factorize(100) == [2, 2, 5, 5]
assert factorize(108) == [2, 2, 3, 3, 3]
assert factorize(120) == [2, 2, 2, 3, 5]
assert factorize(125) == [5, 5, 5]
assert factorize(128) == [2, 2, 2, 2, 2, 2, 2]
assert factorize(144) == [2, 2, 2, 2, 3, 3]
assert factorize(160) == [2, 2, 2, 2, 2, 5]
assert factorize(180) == [2, 2, 3, 3, 5]
assert factorize(200) == [2, 2, 2, 5, 5]
assert factorize(225) == [3, 3, 5, 5]
assert factorize(240) == [2, 2, 2, 2, 3, 5]
assert factorize(250) == [2, 5, 5, 5]
assert factorize(256) == [2, 2, 2, 2, 2, 2, 2, 2]
assert factorize(288) == [2, 2, 2, 2, 2, 3, 3]
assert