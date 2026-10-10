from typing import List

def parse_music(music_string: str) -> List[int]:
    return [ {'o': 4, 'o|': 2, '.|': 1}[tok] for tok in music_string.split() ]
