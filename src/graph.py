##### tree flags
FREE = 0
SOURCE = 1
SINK = 2

################################################################################ NODES
class Node:
    __slots__ = ("pos", "tree", "origin", "parent", "children", "edges", "neighs")
    def __init__(self):
        self.edges = {}
        self.neighs = []

    def __repr__(self):
        return "\n\t".join([
            f"NODE {self.pos}: tree={self.tree}, origin={self.tree}, {len(self.children)} children. Edges:",
            *map(str, self.edges.values())
        ])

    def add_edge(self, head, edge):
        self.edges[head.pos] = edge

    def get_edge(self, head):
        return self.edges[head.pos]

class SourceNode(Node):
    def __init__(self):
        super().__init__()
        self.pos = (-1, -1)
        self.tree = SOURCE
        self.origin = SOURCE

    def __repr__(self): return "SOURCE NODE"

class SinkNode(Node):
    def __init__(self):
        super().__init__()
        self.pos = (-2, -2)
        self.tree = SINK
        self.origin = SINK

    def __repr__(self): return "SINK NODE"

class NonTerminalNode(Node):
    __slots__ = ("parent")
    def __init__(self, pos):
        super().__init__()
        self.pos = pos
        self.tree = FREE
        self.origin = FREE

        self.parent = None # current parent in the search tree
        self.children = [] # current children in the search tree

    def add_neighbor_edge(self, edge):
        self.neighs.append(edge)

    def get_neighbor_edges(self):
        for edge in self.neighs:
            yield edge

    def get_neighbor_nodes(self):
        for edge in self.neighs:
            yield edge.head


# ////////////////////////////////////////////////////////////////////////////// EDGES
class Edge:
    __slots__ = ("tail", "head", "residual")
    def __init__(self, tail, head, weight):
        self.tail = tail
        self.head = head
        self.residual = weight
        self.tail.add_edge(head, self)

    def __repr__(self):
        return f"EDGE {self.tail.pos}->{self.head.pos}: res={self.residual}"

    def get_reverse(self):
        return self.head.get_edge(self.tail)

# ////////////////////////////////////////////////////////////////////////////// GRAPH
class Graph:
    def __init__(self, w, h):
        self.source = SourceNode()
        self.sink = SinkNode()
        self.nodes = [NonTerminalNode((i, j)) for i in range(w) for j in range(h)]

        self.edges_source = []
        self.edges_sink = []

    def add_edge_source(self, head, weight):
        self.edges_source.append( Edge(self.source, head, weight) )

    def add_edge_sink(self, tail, weight):
        self.edges_sink.append( Edge(tail, self.sink, weight) )

    def add_edge_nt(self, tail, head, weight):
        tail.add_neighbor_edge( Edge(tail, head, weight) )
        head.add_neighbor_edge( Edge(head, tail, weight) )


################################################################################
