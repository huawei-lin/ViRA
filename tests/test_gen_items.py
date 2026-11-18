"""vira.eval.gen_items: option parsing of the general benchmarks."""
from vira.eval.gen_items import mmstar_split, strip_letter


def test_strip_letter():
    assert strip_letter("(A) In the upper right area") == "In the upper right area"
    assert strip_letter("B. area of grid B5") == "area of grid B5"
    assert strip_letter("F. No right choice") == "No right choice"


def test_mmstar_split():
    q, o = mmstar_split("Which option describe the object relationship in the image correctly?\nOptions: A: The suitcase "
                        "is on the book., B: The suitcase is beneath the cat., C: beneath, the bed., D: beneath the book.")
    assert q.startswith("Which option") and o == ["The suitcase is on the book.", "The suitcase is beneath the cat.",
                                                  "beneath, the bed.", "beneath the book."]
    q, o = mmstar_split("How many chirality centers does the following molecule have? <image 1>\n"
                        "Options: A: 5, B: Other Answers, C: nan, D: nan")
    assert o == ["5", "Other Answers"]
    q, o = mmstar_split("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the "
                        "end.\nQuestion: What is the sum of 2002, 2003 and 2004?\nChoices:\n(A) 40.7\n(B) 74\n(C) 70.4\n(D) 70.0")
    assert q == "What is the sum of 2002, 2003 and 2004?" and o == ["40.7", "74", "70.4", "70.0"]
    q, o = mmstar_split("Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the "
                        "end.\nQuestion: What percentage of trees have leaves?\n(A) 40\n(B) 50\n(C) 60\n(D) 70")
    assert q == "What percentage of trees have leaves?" and o == ["40", "50", "60", "70"]


def test_letter_split_and_inner_nan():
    from vira.eval.gen_items import letter_split
    assert letter_split("A Fired ceramic pieces\nB Bleached wood fragments\n C Natural stones \nD Smooth concrete casts") == \
        ["Fired ceramic pieces", "Bleached wood fragments", "Natural stones", "Smooth concrete casts"]
    assert letter_split("A. Its mouth is open. B. It is missing its jaw. C. Closed.") == \
        ["Its mouth is open.", "It is missing its jaw.", "Closed."]
    assert letter_split("A. To the left \nB. To the right\nC. Front right\nD. Front left")[3] == "Front left"
    q, o = mmstar_split("How many angels have halos?\nOptions: A: Two, B: One, C: nan, D: Three")
    assert o == ["Two", "One", "nan", "Three"]
