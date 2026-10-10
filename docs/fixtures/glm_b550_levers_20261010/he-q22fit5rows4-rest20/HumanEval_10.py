def make_palindrome(string: str) -> str:
    for i in range(len(string)):
    # Find longest palindromic suffix
    if string[i:] == string[i:][::-1]:
    # Append reverse of prefix before palindromic suffix
    return string[:i] + string[:i][::-1]
