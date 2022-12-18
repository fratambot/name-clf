# Here we have only the modules and libraries necessary for
# the inference API
import re
import unidecode
from string import digits, punctuation


def remove_digits_punctuation_doublespaces(inp):
    no_digits = inp.translate(str.maketrans("", "", digits))
    no_punct = no_digits.translate(str.maketrans("", "", punctuation))
    no_double_spaces = re.sub(" +", " ", no_punct)
    return no_double_spaces.strip()

def unidecode_string(funny_string):
    return unidecode.unidecode(funny_string)
