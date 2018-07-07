import itertools
from BirdsOfAFeatherNode import BirdsOfAFeatherNode
from search import depth_first_search_no_repeats, reset_depth_first_search_no_repeats
from Card import STR_CARD_DICT

stack = []


def play():
    global stack
    print('Enter "?" for command help.  Enter source card and destination card (e.g. "kh qh") to make move.')
    node = BirdsOfAFeatherNode.create_initial()
    reset()
    while True:
        print()
        print(node)
        command = input('? ').strip()
        if command is '?':  # help
            print('[n]ew game, new [s]eed, [h]int, [u]ndo, [a]uto undo, list [f]lockable pairs, list [m]oves, [q]uit')
        elif command is 'n':  # new game
            node = BirdsOfAFeatherNode.create_initial()
            reset()
        elif command is 's':  # new game with specified seed
            seed = int(input('Seed? '))
            node = BirdsOfAFeatherNode.create_initial(seed)
            reset()
        elif command is 'h':  # hint
            if node.is_goal():
                print('Already at goal.')
            elif node.has_separated_flock():
                print('No solution exists from this state.')
            else:
                solvable = False
                for child in node.expand():
                    solvable = depth_first_search_no_repeats(child)
                    if solvable:
                        print('Hint:', child.prev_move)
                        break
                if not solvable:
                    print('No solution exists from this state.')
        elif command is 'u':  # undo
            if stack:
                node = stack.pop()
            else:
                print('Cannot undo.')
        elif command is 'a':  # auto undo to most recent solvable state
            if not stack:
                print('Cannot undo.')
            elif node.is_goal():
                print('Already at goal.')
            else:
                solvable = False
                while not solvable:
                    if not node.has_separated_flock():
                        solvable = depth_first_search_no_repeats(node)
                    if not solvable:
                        if not stack:
                            print('No solution exists for this deal.')
                            break
                        else:
                            node = stack.pop()
        elif command is 'f':  # list flockable pairs
            grid_cards = list(itertools.chain(*node.grid))
            card_list = list(filter(None.__ne__, grid_cards))
            num_cards = len(card_list)
            found_pair = False
            for i in range(num_cards - 1):
                for j in range(i + 1, num_cards):
                    card1 = card_list[i]
                    card2 = card_list[j]
                    if ((card1.get_suit() is card2.get_suit())
                            or (abs(card1.get_rank() - card2.get_rank()) <= 1)):
                        found_pair = True
                        print('{}-{} '.format(card1, card2), end='')
            print('No flockable pairs exist.' if not found_pair else '')
        elif command is 'm':  # list legal moves
            children = node.expand()
            if not children:
                print('No legal moves exist.')
            else:
                for child in children:
                    print('{} '.format(child.prev_move), end='')
                print()
        elif command is 'q':  # quit
            break
        else:  # make a play
            cards = command.upper().split(' ')
            is_legal_play = False
            row1, col1, row2, col2 = -1, -1, -1, -1
            if len(cards) is 2:
                try:
                    card1 = STR_CARD_DICT[cards[0]]
                except KeyError:
                    print('"{}" is not a card.'.format(cards[0]))
                    card1 = None
                try:
                    card2 = STR_CARD_DICT[cards[1]]
                except KeyError:
                    print('"{}" is not a card.'.format(cards[1]))
                    card2 = None
                if card1 and card2:
                    # find rows and columns of cards in grid
                    for r in range(len(node.grid)):
                        for c in range(len(node.grid[0])):
                            card = node.grid[r][c]
                            if card is card1:
                                row1, col1 = r, c
                            if card is card2:
                                row2, col2 = r, c
                    if row1 is -1:
                        print('Card "{}" is not in the grid.'.format(card1))
                    elif row2 is -1:
                        print('Card "{}" is not in the grid.'.format(card2))
                    else:
                        is_legal_play = node.is_legal_move(row1, col1, row2, col2)
            if is_legal_play:
                stack.append(node)
                node = node.make_move(row1, col1, row2, col2)
            else:
                print('Illegal play.')


def reset():
    global stack
    stack = []
    reset_depth_first_search_no_repeats()


if __name__ == '__main__':
    play()
