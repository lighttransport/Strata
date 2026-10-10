def string_xor(a: str, b: str) -> str:
    def xor(i, j):
        return '0' if i == j else '1'
    return ''.join(xor(i, j) for i, j in zip(a, b))
