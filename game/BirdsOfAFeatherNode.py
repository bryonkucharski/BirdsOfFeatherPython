import itertools

import random

import sys



from Card import Card, get_shuffle

from SearchNode import SearchNode





class BirdsOfAFeatherNode(SearchNode):

    grid = [[]]

    prev_move = ''

    def __init__(self):
        super(BirdsOfAFeatherNode,self).__init__()
   
    @classmethod
    def create_initial(cls, seed=random.randint(0, sys.maxsize), rows=4, cols=4):

        """

        Factory method returning an initial Birds of a Feather puzzle node.  The deal seed, if not supplied, will be

        replaced with a random seed.  We also assume 4 rows and columns by default.

        :param seed: specified deal seed

        :param rows: specified rows (default 4)

        :param cols: specified columns (default 4),

        :return: an initial Birds of a Feather puzzle node

        """

        # if not seed:

        #     seed = random.randint(0, sys.maxsize)

        node = BirdsOfAFeatherNode()

        deck = get_shuffle(seed)

        node.grid = [[deck.pop() for _ in range(cols)] for _ in range(rows)]

        return node



    @classmethod

    def from_parent(cls, parent):

        """

        Factory method to create a shallow copy of the parent node with depth one greater and with the given parent

        node.

        :param parent: parent of new child node

        :return: a shallow copy of the parent node with depth one greater and with the given parent node

        """

        child = SearchNode.from_parent(parent)

        child.grid = [[card for card in row] for row in parent.grid]

        return child



    def is_goal(self):

        """

        Return whether or not this is a goal node.  This computation assumes that the grid is rectangular and was

        initially full of cards.

        :return: whether or not this is a goal node

        """

        return self.depth == len(self.grid) * len(self.grid[0]) - 1



    def expand(self):

        """

        Return a list of all children, i.e. successor nodes to this node.

        :return: a list of all children of this node

        """

        children = []

        # for each row pair of cards, collect legal children moves

        for r in range(len(self.grid)):

            for c1 in range(len(self.grid[0]) - 1):

                for c2 in range(c1 + 1, len(self.grid[0])):

                    if self.is_legal_move(r, c1, r, c2):

                        children.append(self.make_move(r, c1, r, c2))

                        children.append(self.make_move(r, c2, r, c1))

        # for each column pair of cards, collect legal children moves

        for c in range(len(self.grid[0])):

            for r1 in range(len(self.grid) - 1):

                for r2 in range(r1 + 1, len(self.grid)):

                    if self.is_legal_move(r1, c, r2, c):

                        children.append(self.make_move(r1, c, r2, c))

                        children.append(self.make_move(r2, c, r1, c))

        return children


    def get_legal_moves(self):
        
        moves = []
        for r in range(len(self.grid)):
            for c1 in range(len(self.grid[0]) - 1):
                for c2 in range(c1 + 1, len(self.grid[0])):
                    if self.is_legal_move(r, c1, r, c2):
                        moves.append([r, c1, r, c2])
                        moves.append([r, c2, r, c1])

        # for each column pair of cards, collect legal children moves

        for c in range(len(self.grid[0])):
            for r1 in range(len(self.grid) - 1):
                for r2 in range(r1 + 1, len(self.grid)):
                    if self.is_legal_move(r1, c, r2, c):
                        moves.append([r1, c, r2, c])
                        moves.append([r2, c, r1, c])

        return moves

    def game_result(self):
        
        if self.is_goal():
            return 1
        elif (len(self.get_legal_moves()) == 0):
            return -1
        else:
            return None

    def is_game_over(self):
        return self.game_result() != None
        
    def is_legal_move(self, row1, col1, row2, col2):

        """

        Return whether or not there is a legal move from [row1][col1] to [row2][col2].

        :param row1: row of source move position

        :param col1: column of source move position

        :param row2: row of destination move position

        :param col2: column of destination move position

        :return: whether or not there is a legal move from [row1][col1] to [row2][col2].

        """

        card1: Card = self.grid[row1][col1]

        card2: Card = self.grid[row2][col2]

        return ((card1 is not None and card2 is not None)  # neither stack empty

                and (card1.get_suit() is card2.get_suit()  # same suit ...

                     or abs(card1.get_rank() - card2.get_rank()) <= 1)  # or same/adjacent rank.

                and ((row1 is row2) or (col1 is col2))  # row or column move

                and (row1 is not row2 or (col1 is not col2)))  # not same cells



    def make_move(self, row1, col1, row2, col2):

        """

        Create a new child node that is the result of this (presumed) legal move from [row1][col1] to [row2][col2].

        :param row1: row of source move position

        :param col1: column of source move position

        :param row2: row of destination move position

        :param col2: column of destination move position

        :return:

        """

        child = BirdsOfAFeatherNode.from_parent(self)

        child.prev_move = repr(child.grid[row1][col1]) + '-' + repr(child.grid[row2][col2])

        child.grid[row2][col2] = child.grid[row1][col1]

        child.grid[row1][col1] = None

        return child



    def __repr__(self):

        """

        Return a string representation of this node as a grid of 2-character card strings or '--' if empty.

        :return: a string representation of this node as a grid of 2-character card strings or '--' if empty.

        """

        return '\n'.join([' '.join(['--' if not card else repr(card) for card in row]) for row in self.grid])



    def solution_string(self):

        """

        Return a string of space-separated moves that solve this deal.

        :return: string of space-separated moves that solve this deal

        """

        moves = []

        node = self

        while node.parent is not None:

            moves.append(node.prev_move)

            node = node.parent

        return ' '.join(list(reversed(moves)))



    def has_odd_bird(self):

        """

        Return whether or not this node has an \"odd bird\", i.e. a card that cannot possibly ever flock to another

        card in the grid.

        :return: whether or not this node has an \"odd bird\"

        """

        grid_cards = list(itertools.chain(*self.grid))

        card_list = list(filter(None.__ne__, grid_cards))

        for card1 in card_list:

            card_is_odd_bird = True

            for card2 in card_list:

                if card2 is card1:

                    continue

                card_is_odd_bird = ((card1.get_suit() is card2.get_suit())

                                    or (abs(card1.get_rank() - card2.get_rank()) <= 1))

                if not card_is_odd_bird:

                    break

            if card_is_odd_bird:

                return True

        return False



    def has_separated_flock(self):

        """

        Return whether or not there exists a group of cards that cannot possible be flocked together with another

        group of cards.  Puzzle heuristics are commonly formed by considering a problem with relaxed constraints.

        In this case, we consider whether or not a flockability graph is a single component, where the flockability

        graph is defined as an undirected graph of card nodes with edges between cards of the same suit or

        same/adjacent rank.  If we find we have separated components, the state is unsolvable and we call this having

        \"separated flocks\".

        :return: whether or not there exists a group of cards that cannot possible be flocked together with another

        group of cards

        """

        grid_cards = list(itertools.chain(*self.grid))

        open_list = list(filter(None.__ne__, grid_cards))

        stack = [open_list.pop()]

        while stack:

            card1 = stack.pop()

            new_open_list = []

            for card2 in open_list:

                if ((card1.get_suit() is card2.get_suit())

                        or (abs(card1.get_rank() - card2.get_rank()) <= 1)):

                    stack.append(card2)

                else:

                    new_open_list.append(card2)

            open_list = new_open_list

        return True if open_list else False





def test_random_walk():

    # Random walk from random deal.

    node = BirdsOfAFeatherNode.create_initial()

    while node is not None:

        print('Current:')

        print(repr(node))

        print()

        children = node.expand()

        print('Children:')

        for child in children:

            print('Move:', child.prev_move)

            print(repr(child))

            print()

        # Random move if possible:

        node = None if not children else random.choice(children)

        print('***********************************************')





if __name__ == '__main__':

    test_random_walk()

