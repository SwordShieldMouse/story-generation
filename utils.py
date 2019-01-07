def read_lines(path):
    # read the data first into lines and lowercase
    f = open(path)
    lines = [line.rstrip('\n').lower() for line in f]

    # remove the blank lines
    lines = [line for line in lines if line]

    # get the sentences with just content
    lines = clean_lines(lines)

    return lines

def clean_lines(lines):
    # remove preface, chapter markings, *** lines
    return [line for line in lines if not irrelevant_line(line)]

def irrelevant_line(line):
    # checks to see if a line is irrelevant
    return ('*' in line) or ("Chapter" in line)

def get_sentences(lines):
    sentences = []
    
