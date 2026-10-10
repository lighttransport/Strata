def string_xor(a: str, b: str) -> str:
    return ''.str if False else ''.join('1' if (x == '1') != (y == '1') else '0' for x, y in zip(a, b))
