from abc import ABC, abstractmethod
import copy


class SearchNode(ABC):
    parent = None

    def __init__(self, depth=0, parent=None):
        self.depth = depth
        self.parent = parent

    @classmethod
    def from_parent(cls, parent):
        """Factory method to create a shallow copy of the parent node with depth one greater and with this node as
        parent."""
        child = copy.copy(parent)  # shallow copy
        child.depth = parent.depth + 1
        child.parent = parent
        return child

    @abstractmethod
    def is_goal(self):
        """Return whether or not this is a goal node."""
        return False

    @abstractmethod
    def expand(self):
        """Return a list of children, i.e. successor nodes to this node."""
        return []
