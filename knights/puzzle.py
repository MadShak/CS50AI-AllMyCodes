from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# General rules for every puzzle:
# 1. A person is either a Knight or a Knave, but not both.
base_rules_a = And(Or(AKnight, AKnave), Not(And(AKnight, AKnave)))
base_rules_b = And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))
base_rules_c = And(Or(CKnight, CKnave), Not(And(CKnight, CKnave)))

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    base_rules_a,
    # If A is a knight, the statement is true; if knave, it's false
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    base_rules_a, base_rules_b,
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    base_rules_a, base_rules_b,
    # A's statement: (AKnight and BKnight) or (AKnave and BKnave)
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    # B's statement: (BKnight and AKnave) or (BKnave and AKnight)
    Biconditional(BKnight, Or(And(BKnight, AKnave), And(BKnave, AKnight)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave.'"
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    base_rules_a, base_rules_b, base_rules_c,

    # A said either "I am a knight" or "I am a knave"
    # If A is a knight, one of those must be true.
    # Note: Biconditional(AKnight, AKnight) is a tautology, but we represent the speech:
    Or(Biconditional(AKnight, AKnight), Biconditional(AKnight, AKnave)),

    # B says "A said 'I am a knave'"
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),

    # B says "C is a knave"
    Biconditional(BKnight, CKnave),

    # C says "A is a knight"
    Biconditional(CKnight, AKnight)
)
