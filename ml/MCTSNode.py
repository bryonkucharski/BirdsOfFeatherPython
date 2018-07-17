import numpy as np
from collections import defaultdict

class MonteCarloTreeSearchNode:
    
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = []
        self._number_of_visits = 0.
        self._results = defaultdict(int)

    def __repr__(self):
        return self.state.__repr__()

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_moves()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def mcts_expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.make_move(action[0],action[1],action[2],action[3])
        child_node = MonteCarloTreeSearchNode(next_state, parent = self)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_moves() #this should return a list of [[r1,c1,r2,c2],[r1,c1,r2,c2], . . . etc]
            
            #print(current_rollout_state)
            #print(current_rollout_state.is_game_over())
            #print(current_rollout_state.game_result() )
            #print(current_rollout_state.is_goal(), current_rollout_state.has_odd_bird(),current_rollout_state.has_separated_flock())

            action = self.rollout_policy(possible_moves) #this shoudl return a single list which is one move
            current_rollout_state = current_rollout_state.make_move(action[0],action[1],action[2],action[3])

        
        return current_rollout_state.game_result() 

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_terminal_node(self):
       # return self.state.is_goal() or self.state.has_odd_bird() or self.state.has_separated_flock()
       return self.state.is_game_over()

    def is_mcts_goal(self):
        return self.state.is_goal()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param = 1.4):
        choices_weights = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ]

        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(0,len(possible_moves))]