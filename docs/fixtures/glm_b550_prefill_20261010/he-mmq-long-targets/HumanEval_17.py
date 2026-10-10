from typing import List

def parse_music(music_string: str) -> List[int]:
    out = []
    i = 0
    while i < len(music_string):
        if music_string.startswith('o|', i):
            out.append(2); i += 2
        elif music_string.startswith('.|', i):
            out.append(1); i += 2
        elif music_string[i] == 'o':
            out.append(4); i += 1
        else:
            i += 1
    return out
