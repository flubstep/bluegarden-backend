class Material500(object):
    RED = 0xf44336
    PINK = 0xe81e62
    PURPLE = 0x9c27b0
    DEEP_PURPLE = 0x673ab7
    INDIGO = 0x3f51b5
    BLUE = 0x2196f3
    LIGHT_BLUE = 0x03a9f4
    CYAN = 0x00bcd4
    TEAL = 0x009688
    GREEN = 0x4caf50
    LIGHT_GREEN = 0x8bc34a
    LIME = 0xcddc39
    YELLOW = 0xffeb3b
    AMBER = 0xffc107
    ORANGE = 0xff9800
    DEEP_ORANGE = 0xff5722
    BROWN = 0x795548
    GREY = 0x9e9e9e
    BLUE_GREY = 0x607d8b
    WHITE = 0xffffff

def hex_to_rgb(hex):
    R = (hex & 0xff0000) >> 16
    G = (hex & 0x00ff00) >> 8
    B = hex & 0x0000ff
    return (R, G, B)

CLASSIFICATION_TO_COLOR = {
    1: Material500.GREY, # Unclassified
    2: Material500.BROWN, # Bare earth
    7: Material500.GREY, # Low noise
    9: Material500.LIGHT_BLUE, # Water
    10: Material500.BROWN, # Ignored ground
    11: Material500.GREY, # Withheld,
    17: Material500.BLUE_GREY, # Bridge decks
    18: Material500.ORANGE # High noise
}

def rgb_for_classification(classification):
    color = CLASSIFICATION_TO_COLOR.get(classification, Material500.AMBER)
    return hex_to_rgb(color)
