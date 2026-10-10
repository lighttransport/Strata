def string_xor(a: str, b: str) -> str:
    return ''.join('0' if x == y else '1' for x, y in zip(a, b))
