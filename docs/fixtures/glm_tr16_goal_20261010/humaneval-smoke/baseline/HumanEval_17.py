from typing import List

def parse_music(music_string: str) -> List[int]:
    legend = {'o': 4, 'o|': 2, '.|': 1}
    return [legend[note] for note in music_string.split()]
