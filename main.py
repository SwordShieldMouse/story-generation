from architectures import *
from utils import *

path = 'data/pride-and-prejudice.txt'

# first try this with a limited dataset (i.e., choose a book)
lines = read_lines(path)
sentences = get_sentences(lines)
