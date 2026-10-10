def parse_music(music_string: str) -> List[int]:
    mapping = {'o': 4, 'o|': 2, '.|': 1}
    return [mapping[t] for t in music_string.split()]
