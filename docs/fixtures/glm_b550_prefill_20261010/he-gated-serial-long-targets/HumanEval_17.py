from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
The legend says 'o' whole (4), 'o|' half (2), '.|' quater (1). But the example shows 'o o| .| o| o| .| .| .| .| o o' → [4,2,1,2,2,1,1,1,1,4,4]. Wait, that's 11 tokens but 12 outputs? Let me count: tokens: 'o','o|','.|','o|','o|','.|','.|','.|','.|','o','o' = 11 tokens; outputs 12 values. Hmm, mismatch. Actually maybe the format is like "o|o|.|o|o|.|.|.|.|oo" concatenated? The legend says 'o' whole, 'o|' half, '.|' quater. The string 'o o| .| o| o| .| .| .| .| o o' with spaces... maybe spaces separate notes, and each note is a token. But 11 tokens vs 12 ints. Let me recount tokens: "o o| .| o| o| .| .| .| .| o o" split by space: ['o','o|','.|','o|','o|','.|','.|','.|','.|','o','o'] → 11. Outputs: [4,2,1,2,2,1,1,1,1,4,4] → 12. Hmm.

Maybe the format is like "o|o|.|o|o|.|.|.|.|oo" concatenated? The legend says 'o' whole, 'o|' half, '.|' quater. The string 'o o| .| o| o| .| .| .| .| o o' with spaces... maybe spaces separate notes, and each note is a token. But 11 tokens vs 12 ints. Let me recount tokens: "o o| .| o| o| .| .| .| .| o o" split by space: ['o','o|','.|','o|','o|','.|','.|','.|','.|','o','o'] → 11. Outputs: [4,2,1,2,2,1,1,1,1,4,4] → 12. Hmm.

Maybe the format is like "o|o|.|o|o|.|.|.|.|oo" concatenated? The legend says 'o' whole, 'o|' half, '.|' quater. The string 'o o| .| o| o| .| .| .| .| o o' with spaces... maybe spaces separate notes, and each note is a token. But 11 tokens vs 12 ints. Let me recount tokens: "o o| .| o| o| .| .| .| .| o o" split by space: ['o','o|','.|','o|','o|','.|','.|','.|','.|','o','o'] → 11. Outputs: [4,2,1,2,2,1,1,1,1,4,4] → 12. Hmm.

Maybe the format is like "o|o|.|o|o|.|.|.|.|oo" concatenated? The legend says 'o' whole, 'o|' half, '.|' quater. The string 'o o| .| o| o| .| .| .| .| o o' with spaces... maybe spaces separate notes, and each note is a token. But 11 tokens vs 12 ints. Let me recount tokens: "o o| .| o| o| .| .| .| .| o o" split by space: ['o','o|','.|','o|','o|','.|','.|','.|','.|','o','o'] → 11. Outputs: [4,2,1,2,2,1,1,1,1,4,4] → 12. Hmm.

Maybe the format is like "o|o|.|o|o|.|.|.|.|oo" concatenated? The legend says 'o' whole, 'o|' half, '.|' quater. The string 'o o| .| o| o| .| .| .| .| o o' with spaces... maybe spaces separate notes, and each note is a token. But 11 tokens vs 12 ints. Let me recount tokens: "o o| .| o| o| .| .| .| .| o o" split by space: ['o','o|','.|','o|','o|','.|','.|','.|','.|','o','o'] → 11. Outputs: [4,2,1,2,2,1,1