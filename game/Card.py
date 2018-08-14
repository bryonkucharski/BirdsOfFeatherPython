import freecell_shuffler

# 'RANK_NAMES' array of abbreviated card rank names in ascending order of rank and indexed by suit index
RANK_NAMES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
# 'SUIT_NAMES' array of abbreviated card suit names indexed by suit index
SUIT_NAMES = ['C', 'H', 'S', 'D']
# 'NUM_RANKS' number of card ranks
NUM_RANKS = len(RANK_NAMES)
# 'NUM_SUITS' number of card suits
NUM_SUITS = len(SUIT_NAMES)
# 'NUM_CARDS' number of cards
NUM_CARDS = NUM_RANKS * NUM_SUITS


class Card:
    """A class for representing standard (French) playing cards.
    Rank numbers are 0 through 12, corresponding to rank names: 'A','2','3','4','5','6','7','8','9','T','J','Q','K'.
    Suit numbers are 0 through 3, corresponding to suit names: 'C','H','S','D'.
    The String representation for each card will be the concatenation of a rank name with a suit name.

    It's possible to go between 4 different card representations using this class:
    (1) Card object representation
    (2) String representation
    (3) single integer representation (0 - 51)
    (4) two integer (rank, suit) representation

    Avoid the construction of new Card objects. Use the Card objects already created in ALL_CARDS, retrieving them via
    (1) dictionary STR_CARD_DICT lookup by two-character String representation
    (2) function get_card(int), or
    (3) function get_card(int rank, int suit).
    """
    def __init__(self, rank, suit):
        """Initialize a card with given rank and suit 0-based indices."""
        self.__rank = rank
        self.__suit = suit

    def get_rank(self):
        """Return the 0-based rank index of the Card."""
        return self.__rank

    def get_suit(self):
        """Return the 0-based suit index of the Card."""
        return self.__suit

    def get_id(self):
        """Return the 0-based id of the Card."""
        return self.__suit * NUM_RANKS + self.__rank

    def __repr__(self):
        """Return the 2-character rank-and-suit string representation of the Card. """
        return RANK_NAMES[self.__rank] + SUIT_NAMES[self.__suit]


# 'ALL_CARDS' array mapping card ids to Card objects
ALL_CARDS = []
# 'STR_CARD_DICT' dictionary mapping repr(Card) to associated Card object
STR_CARD_DICT = {}
# 'STR_ID_DICT' dictionary mapping repr(Card) to associated Card id
STR_ID_DICT = {}
# 'ID_STR_DICT' dictionary mapping Card id to associated repr(Card)
ID_STR_DICT = {}

# Initialize ALL_CARDS, STR_CARD_DICT, STR_ID_DICT, ID_STR_DICT
for s in range(NUM_SUITS):
    for r in range(NUM_RANKS):
        c = Card(r, s)
        ALL_CARDS.append(c)
        STR_CARD_DICT[repr(c)] = c
        STR_ID_DICT[repr(c)] = c.get_id()
        ID_STR_DICT[c.get_id()] = repr(c)


def id_to_card(id_num):
    """Return the Card associated with the given 0-based id."""
    return ALL_CARDS[id_num]


def get_card(rank, suit):
    """Return the Card with the given rank and suit indices.  This is preferred to constructing Card objects,
    as all 52 Card objects are already created and there is no need to create duplicates."""
    return ALL_CARDS[suit * NUM_RANKS + rank]


def rank_suit_to_id(rank, suit):
    """Return the 0-based id number associated with the given 0-based rank and suit indices."""
    return suit * NUM_RANKS + rank


def get_shuffle(seed):
    """Return a list of Cards (to be treated as a stack) that is associated with the given Microsoft
     FreeCell deal seed."""
    return [STR_CARD_DICT[card_str] for card_str in freecell_shuffler.get_deck_str(seed)]
