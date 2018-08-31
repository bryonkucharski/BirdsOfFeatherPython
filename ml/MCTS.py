import os, sys
import pdb
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name + "/game")

from BirdsOfAFeatherNode import BirdsOfAFeatherNode
from MCTSNode import MonteCarloTreeSearchNode

class MonteCarloTreeSearch:
    
    def __init__(self, node):
        self.root = node



    def best_action(self, simulations_number):
        for i in range(0, simulations_number):      
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            print(reward)
        # exploitation only
        return self.root.best_child(c_param = 0.)


    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.mcts_expand()
            else:
                current_node =  current_node.best_child()
              
        return current_node

#state = np.zeros((3,3))
#initial_board_state = TicTacToeGameState(state = state, next_to_move = 1)
'''
inital_state = BirdsOfAFeatherNode.create_initial(0)  # (367297990)
root = MonteCarloTreeSearchNode(state = inital_state, parent = None)
mcts = MonteCarloTreeSearch(root)
best_node = mcts.best_action(1600)
print(best_node, best_node.q, best_node.n)
'''
