def is_palindrome(string: str) -> bool:
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
    return string

# Tests
assert make_palindrome('') == ''
assert make_palindrome('cat') == 'catac'
assert make_palindrome('cata') == 'catac'
assert make_palindrome('cats') == 'catscat'