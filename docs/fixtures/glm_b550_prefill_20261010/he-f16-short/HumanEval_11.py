def string_xor(a: str, b: str) -> str:
    return ''.join('1' if (i == '1') ^ (j == '1') else '0' for i, j in zip(a, b))
