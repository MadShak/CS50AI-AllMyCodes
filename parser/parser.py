import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "but"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# =============================================================================
# GRAMMAR RULES – NO INLINE COMMENTS
# =============================================================================
NONTERMINALS = """
S -> NP
S -> NP VP
S -> S Conj S

NP -> N
NP -> Det N
NP -> Det AdjP N
NP -> Det N N
NP -> N Adj
NP -> NP PP
NP -> NP AdvP
NP -> NP Conj NP

AdjP -> Adj
AdjP -> Adj AdjP

AdvP -> Adv
AdvP -> Adv AdvP

PP -> P NP

VP -> V
VP -> V NP
VP -> V PP
VP -> V NP PP
VP -> VP Conj VP
VP -> Adv VP
VP -> VP Adv
"""

# Build the grammar once, reuse the parser everywhere
GRAMMAR = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(GRAMMAR)


def preprocess(sentence):
    """
    Lowercase, tokenize, and keep only words with at least one letter.
    """
    tokens = nltk.word_tokenize(sentence)
    words = []
    for token in tokens:
        low = token.lower()
        if any(ch.isalpha() for ch in low):
            words.append(low)
    return words


def np_chunk(tree):
    """
    Return all NP subtrees that do not contain any other NP inside them.
    """
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            # check if any descendant (excluding itself) is also an NP
            has_inner_np = any(
                child.label() == 'NP'
                for child in subtree.subtrees()
                if child != subtree
            )
            if not has_inner_np:
                chunks.append(subtree)
    return chunks


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    elif len(sys.argv) == 1:
        s = input("Sentence: ")
    else:
        sys.exit("Usage: python parser.py [filename]")

    words = preprocess(s)

    # make sure punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    trees = list(parser.parse(words))
    if not trees:
        sys.exit("No parse tree found.")

    for tree in trees:
        tree.pretty_print()
        print("Noun Phrase Chunks")
        chunks = np_chunk(tree)
        for chunk in chunks:
            print(" ".join(chunk.leaves()))


if __name__ == "__main__":
    main()
