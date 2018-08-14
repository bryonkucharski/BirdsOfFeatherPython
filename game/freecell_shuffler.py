from sys import argv
"""
based on http://rosettacode.org/wiki/Deal_cards_for_FreeCell#Python
Updated to Python 3 by Todd W. Neller May 17, 2018.
TWN reversed deal order to follow stack convention for dealing of cards from top (last) index
TWN added string get_deck_str(int seed) for use by Card method: get_shuffle(int seed)

NOTE: You will not need to use this code directly. Instead use the Card getShuffle(int seed) function
"""


def random_generator(random_seed=1):
    max_int32 = (1 << 31) - 1
    random_seed = random_seed & max_int32
    while True:
        random_seed = (random_seed * 214013 + 2531011) & max_int32
        yield random_seed >> 16


def deal(random_seed):
    num_cards = 52
    cards = list(range(num_cards - 1, -1, -1))
    rnd = random_generator(random_seed)
    for i, r in zip(range(num_cards), rnd):
        j = (num_cards - 1) - r % (num_cards - i)
        cards[i], cards[j] = cards[j], cards[i]
    return list(reversed(cards))


def show(cards):
    reversed_deck = ["A23456789TJQK"[c // 4] + "CDHS"[c % 4] for c in reversed(cards)]
    for i in range(0, len(cards), 8):
        print(" ", " ".join(reversed_deck[i: i + 8]))


def get_deck_str(random_seed):
    return ["A23456789TJQK"[c // 4] + "CDHS"[c % 4] for c in deal(random_seed)]


if __name__ == '__main__':
    seed = int(argv[1]) if len(argv) == 2 else 617
    deck = deal(seed)
    show(deck)
  

