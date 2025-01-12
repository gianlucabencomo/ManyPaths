import numpy as np

MLP_PARAMS = [
    (8, 2),  # 0
    (16, 2),  # 1
    (32, 2),  # 2
    (64, 2),  # 3
    (8, 4),  # 4
    (16, 4),  # 5
    (32, 4),  # 6
    (64, 4),  # 7
    (8, 6),  # 8
    (16, 6),  # 9
    (32, 6),  # 10
    (64, 6),  # 11
    (8, 8),  # 12
    (16, 8),  # 13
    (32, 8),  # 14
    (64, 8),  # 15
]

CNN_PARAMS = [
    ([16, 8], 2),  # 0
    ([32, 16], 2),  # 1
    ([64, 32], 2),  # 2
    ([128, 64], 2),  # 3
    ([16, 8, 4, 2], 4),  # 4
    ([32, 16, 8, 4], 4),  # 5
    ([64, 32, 16, 8], 4),  # 6
    ([128, 64, 32, 16], 4),  # 7
    ([16, 8, 4, 2], 6),  # 8
    ([32, 16, 8, 4], 6),  # 9
    ([64, 32, 16, 8], 6),  # 10
    ([128, 64, 32, 16], 6),  # 11
    ([16, 8, 4, 2], 8),  # 12
    ([32, 16, 8, 4], 8),  # 13
    ([64, 32, 16, 8], 8),  # 14
    ([128, 64, 32, 16], 8),  # 15
]

LSTM_PARAMS = [
    (8, 1),  # 0
    (16, 1),  # 1
    (32, 1),  # 2
    (64, 1),  # 3
    (8, 2),  # 4
    (16, 2),  # 5
    (32, 2),  # 6
    (64, 2),  # 7
    (8, 3),  # 8
    (16, 3),  # 9
    (32, 3),  # 10
    (64, 3),  # 11
    (8, 4),  # 12
    (16, 4),  # 13
    (32, 4),  # 14
    (64, 4),  # 15
]

TRANSFORMER_PARAMS = [
    (8, 1),  # 0
    (16, 1),  # 1
    (32, 1),  # 2
    (64, 1),  # 3
    (8, 2),  # 4
    (16, 2),  # 5
    (32, 2),  # 6
    (64, 2),  # 7
    (8, 3),  # 8
    (16, 3),  # 9
    (32, 3),  # 10
    (64, 3),  # 11
    (8, 4),  # 12
    (16, 4),  # 13
    (32, 4),  # 14
    (64, 4),  # 15
]

MODELS = ["mlp", "transformer", "lstm", "cnn"]
DATA_TYPES = ["number", "bits", "image"]
SKIPS = [1, 2]  # 20-20 and even-odd
INDICES = np.arange(16)
DEFAULT_INDEX = 7

FEATURE_VALUES = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
)

ALPHABETS = {
    "misc" : ["Alphabet_of_the_Magi", "Braille", "Futurama"],
    "ancient" : ["Anglo-Saxon_Futhorc", "Arcadian", "Armenian", "Asomtavruli_(Georgian)", "Early_Aramaic", "Grantha", "Greek", "Hebrew", "Latin", "Sanskrit", "Syriac_(Estrangelo)", "Tifinagh"],
    "asian" : ["Balinese", "Bengali", "Burmese_(Myanmar)", "Grantha", "Gujarati", "Japanese_(hiragana)", "Japanese_(katakana)", "Korean", "Malay_(Jawi_-_Arabic)", "Sanskrit", "Tagalog"],
    "european" : ["Anglo-Saxon_Futhorc", "Cyrillic", "Greek", "Braille", "Latin"],
    "middle" : ["Armenian", "Asomtavruli_(Georgian)", "Early_Aramaic", "Hebrew", "Malay_(Jawi_-_Arabic)", "Mkhedruli_(Georgian)", "Syriac_(Estrangelo)"],
    "aboriginal" : ["Blackfoot_(Canadian_Aboriginal_Syllabics)", "Inuktitut_(Canadian_Aboriginal_Syllabics)", "Ojibwe_(Canadian_Aboriginal_Syllabics)"],
    "african" : ["N_Ko", "Tifinagh"]
}