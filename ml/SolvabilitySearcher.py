class SolvabilitySearcher():

    '''
	 * <code>search</code> - given an initial node, perform recursive
	 * depth-first search (DFS), caching (un)solvable node Strings
	 *
	 * @param node a <code>SearchNode</code> value - the node to be searched
	 * @return a <code>boolean</code> value - whether or not goal node
	 * was found 
     '''
    def __init__(self):
        self.solvable = []
        self.unsolvable = []
        self.nodeCount = 0
        self.goal_node = None

    def search(self,node):
        self.nodeCount += 1
        node_str = node.__repr__()
        if node_str in self.solvable:
            return True
        if node_str in self.unsolvable:
            return False
        if node.is_goal():
            self.goal_node = node
            self.solvable.append(node_str)
            return True
        else:
            children = node.expand()
            for child in children:
                if(self.search(child)):
                    self.solvable.append(node_str)
                    return True
            self.unsolvable.append(node_str)
            return False