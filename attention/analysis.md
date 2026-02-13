# Attention Analysis – CS50 AI

I ran `mask.py` on several sentences and examined the 144 attention diagrams.

---

## Head 5, Layer 7 – Determiners attend to their nouns

This head consistently shows that determiners ("a", "the", "my") attend strongly to the noun they modify.

**Example 1:**
Sentence: *"The [MASK] barked loudly."*
In the diagram, token "the" (row) has its brightest cell under the column for "barked". The determiner is looking forward to the noun.

**Example 2:**
Sentence: *"I saw an [MASK] in the sky."*
Token "an" attends most to the masked token (which BERT correctly predicted as "airplane"). Even without knowing the word, the determiner anchors to the upcoming noun.

---

## Head 2, Layer 9 – Pronouns attend to their antecedents

This head appears to track coreference: pronouns look back at the noun they refer to.

**Example 1:**
Sentence: *"Holmes picked up his [MASK]."*
Token "his" (row) has its brightest attention on "Holmes" (column). The possessive pronoun points to the owner.

**Example 2:**
Sentence: *"When Mary arrived, she was [MASK]."*
Token "she" attends most heavily to "Mary". The subject pronoun links to the named person earlier in the sentence.

Neither of these patterns is perfect in every sentence, but the tendency is clear across multiple examples.
