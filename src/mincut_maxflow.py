import os, time, psutil
import numpy as np
from collections import Counter

from graph import Graph, FREE, SOURCE, SINK

################################################################################
def region_to_borders(region : np.array):
    """
    Input: binary array:
        0: exterior
        1: interior and borders of the regions
    Output: binary array:
        0: exterior and interior of the regions
        1: borders of the regions
    """

    center = region[1:-1, 1:-1]
    left = region[1:-1, :-2]
    right = region[1:-1, 2:]
    top = region[:-2, 1:-1]
    bottom = region[2:, 1:-1]

    horizontal = np.logical_or(
        np.logical_xor(center, left),
        np.logical_xor(center, right)
    )
    vertical = np.logical_or(
        np.logical_xor(center, top),
        np.logical_xor(center, bottom)
    )
    mask = np.logical_or(horizontal, vertical)
    region[1:-1, 1:-1] = np.logical_and(center, mask)
    return region

def benchmark(init_time, process):
    print(f"...>>> Elapsed time: {time.time() - init_time:.2f} seconds")
    print(f"...>>> Memory used: {process.memory_info().rss / 1e6:.1f} MB")

##### default cut parameters
CYCLES_PER_FRAME = 10 # amount of cut cycles to perform before updating the GUI (Animate modes)
SIZE_RANDOM = 100 # size of the random array (RND modes)

# //////////////////////////////////////////////////////////////////////////////
class GraphCut:
    def __init__(self, img, seeds, cut_sigma, cut_lambda):
        self.process = psutil.Process(os.getpid())
        init_time = time.time()

        ######################################################################## INITIALIZATION
        self.cut_sigma = cut_sigma
        self.cut_lambda = cut_lambda

        self.img = img                  # matrix with Ip information of the original image's pixels
        self.img_norm = self.img / 255
        self.w, self.h = self.img.shape

        self.seeds_obj = seeds[:,:,0] != 0 # boolean matrix, True for red pixels (object seed) --> source S
        self.seeds_bkg = seeds[:,:,2] != 0 # boolean matrix, True for blue pixels (background seed) --> sink T

        self.graph = Graph(self.w, self.h)
        self.A_nodes = [] # active nodes
        self.O_nodes = [] # orphan nodes
        self.path_ST = []

        #####################
        self.node_isactive = np.zeros(self.img.shape, dtype = bool)
        self.TREE = np.zeros(self.img.shape, dtype = np.uint8)

        self.overlay = np.zeros((self.w, self.h, 4), dtype = np.uint8)
        self.mat_path_ST = np.zeros((self.w, self.h, 4), dtype = np.uint8)

        #####################
        self.randoms = np.random.random_integers(low = 0, high = SIZE_RANDOM, size = SIZE_RANDOM)
        self.random_i = 0
        self.cut_cycle = 0

        self.compute_hists()
        self.init_nonterminal_edges()
        self.init_terminal_edges()

        self.node_isactive = region_to_borders(self.node_isactive)
        border_nodes = np.nonzero(self.node_isactive)
        self.A_nodes = [self.graph.nodes[i * self.h + j] for i,j in zip(*border_nodes)]

        print(f">>> Parameters: Sigma = {self.cut_sigma}, Lambda = {self.cut_lambda}")
        print(f">>> Initialization finished.")
        benchmark(init_time, self.process)


    def penalty_Rp(self, probability, pixel_intensity):   # for terminal edges
        prob_given_seed = probability[pixel_intensity]
        penalty = -np.log(prob_given_seed) if prob_given_seed else self.K
        return self.cut_lambda * penalty


    def penalty_Bp(self, pi_0, pi_1):                     # for non-terminal edges
        penalty = np.exp(-((pi_0 - pi_1) ** 2) / (2 * (self.cut_sigma ** 2)))
        return penalty


    def compute_hists(self):
        self.OBJ_pdf = np.zeros(256)
        self.BKG_pdf = np.zeros(256)

        obj = self.img[self.seeds_obj]
        bkg = self.img[self.seeds_bkg]

        for Ip, count in Counter(obj).items(): self.OBJ_pdf[Ip] = count
        for Ip, count in Counter(bkg).items(): self.BKG_pdf[Ip] = count

        self.BKG_pdf /= len(bkg)
        self.OBJ_pdf /= len(obj)



    def init_terminal_edges(self):
        for node in self.graph.nodes:
            if self.seeds_obj[node.pos]: # pixel is part of foreground seed
                self.graph.add_edge_source(node, self.K)
                self.graph.add_edge_sink(node, 0)

                node.parent = self.graph.source
                self.set_tree(node, SOURCE)
                node.origin = SOURCE
                self.node_isactive[node.pos] = True

            elif self.seeds_bkg[node.pos]: # pixel is part of background seed
                self.graph.add_edge_source(node, 0)
                self.graph.add_edge_sink(node, self.K)

                node.parent = self.graph.sink
                self.set_tree(node, SINK)
                node.origin = SINK
                self.node_isactive[node.pos] = True

            else: # pixel isn't seeded
                self.graph.add_edge_source(node, self.penalty_Rp(self.BKG_pdf, self.img[node.pos]))
                self.graph.add_edge_sink  (node, self.penalty_Rp(self.OBJ_pdf, self.img[node.pos]))



    def init_nonterminal_edges(self):
        self.K = 0
        for node_0 in self.graph.nodes:
            i,j = node_0.pos
            self.sum_Bp = 0
            if i > 0:
                self.init_nt_edge(node_0, self.graph.nodes[(i - 1) * self.h + j])
            if i < self.w - 1:
                self.init_nt_edge(node_0, self.graph.nodes[(i + 1) * self.h + j])
            if j > 0:
                self.init_nt_edge(node_0, self.graph.nodes[i * self.h + j - 1])
            if j < self.h - 1:
                self.init_nt_edge(node_0, self.graph.nodes[i * self.h + j + 1])
            if self.sum_Bp > self.K:
                self.K = self.sum_Bp
        self.K += 1


    ########################################################################

    def init_nt_edge(self, node_0, node_1):
        penalty = self.penalty_Bp(self.img_norm[node_0.pos], self.img_norm[node_1.pos])
        self.graph.add_edge_nt(node_0, node_1, penalty)
        self.sum_Bp += penalty

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    ############################################################################ ITERATION STEPS
    def grow(self): # stage A
        while self.A_nodes: ### while A =/= ∅
            p = self.pick_active_node() ### pick an active node p ∈ A

            for q in p.get_neighbor_nodes(): ### for every neighbor q ...
                if self.tree_cap(p, q) > 0: ### ... such that tree cap(p → q) > 0
                    if q.tree == FREE: ### if TREE(q) = ∅
                        self.make_child(p, q) ### PARENT(q) := p
                        self.set_tree(q, p.tree) ### TREE(q) := TREE(p)
                        q.origin = p.origin

                        self.A_nodes.append(q) ### A := A ∪ {q}
                        self.node_isactive[q.pos] = True

                    elif q.tree != p.tree: ### if TREE(q) =/= ∅ and TREE(q) =/= TREE(p)
                        return self.construct_path_ST(p, q)

            self.remove_active_node() ### remove p from A
            self.node_isactive[p.pos] = False

        self.path_ST.clear()


    def augment(self): # stage B
        ### find the bottleneck capacity ∆ on P
        bottleneck_capacity = min(self.path_ST, key = lambda edge: edge.residual).residual

        for edge in self.path_ST: ### update the residual graph by pushing flow ∆ through P
            edge.residual -= bottleneck_capacity

        for edge in self.path_ST:
            if not edge.residual: ### for each edge (p, q) in P that becomes saturated
                p,q = edge.tail, edge.head
                if p.tree == q.tree == SOURCE:
                    self.make_orphan(q)
                    self.change_branch_origin(q, FREE)
                elif p.tree == q.tree == SINK:
                    self.make_orphan(p)
                    self.change_branch_origin(p, FREE)


    def adopt(self): # stage C
        while self.O_nodes:
            orphan = self.O_nodes.pop()
            succesful_adoption = False

            ##### looking for potential adoption by neighbors
            for neigh in orphan.get_neighbor_nodes():
                if orphan.tree != neigh.tree: continue
                if neigh.origin is FREE: continue
                if self.tree_cap(neigh, orphan) > 0:
                    succesful_adoption = True; break

            if succesful_adoption:
                self.make_child(neigh, orphan)
                self.change_branch_origin(orphan, neigh.tree)
                continue

            for neigh in orphan.get_neighbor_nodes():
                if orphan.tree != neigh.tree: continue
                if self.tree_cap(neigh, orphan) > 0:
                    if not self.node_isactive[neigh.pos]:
                        self.A_nodes.append(neigh)
                        self.node_isactive[neigh.pos] = True

                if neigh.parent is orphan:
                    self.make_orphan(neigh)

            self.set_tree(orphan, FREE)

            if self.node_isactive[orphan.pos]:
                self.A_nodes.remove(orphan)
                self.node_isactive[orphan.pos] = False


    def end_cut(self):
        print(f">>> Depleted active nodes after {self.cut_cycle} cut cycles.")
        benchmark(self.start_time, self.process)
        return False


    ############################################################################
    def set_tree(self, node, tree):
        node.tree = tree
        self.TREE[node.pos] = tree

    def tree_cap(self, p, q):
        if p.tree is SOURCE:
            return p.get_edge(q).residual
        return q.get_edge(p).residual

    def construct_path_ST(self, p, q):
        forward, reverse = (p, q) if (p.origin is SOURCE) else (q, p)
        self.path_ST = [forward.get_edge(reverse)]

        while forward is not self.graph.source:
            path_parent = forward.parent
            edge = path_parent.get_edge(forward)
            self.path_ST.insert(0, edge)
            forward = path_parent

        while reverse is not self.graph.sink:
            path_child = reverse.parent
            edge = reverse.get_edge(path_child)
            self.path_ST.append(edge)
            reverse = path_child


    def make_child(self, to_be_parent, to_be_child):
        to_be_child.parent = to_be_parent
        to_be_parent.children.append(to_be_child)

    def make_orphan(self, to_be_orphan):
        to_be_orphan.parent.children.remove(to_be_orphan)
        to_be_orphan.parent = None
        self.O_nodes.append(to_be_orphan)

    def change_branch_origin(self, branch_root, new_origin):
        branch = [branch_root]
        while branch:
            node = branch.pop()
            node.origin = new_origin
            branch += node.children

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ DISPLAY
    def output_array(self):
        self.overlay[:,:,0] = (self.TREE == SOURCE).astype(int) * 255
        self.overlay[:,:,2] = (self.TREE == SINK).astype(int) * 255
        self.overlay[:,:,3] = (self.TREE != FREE).astype(int) * 128
        return self.overlay

    def output_array_BW(self):
        return self.TREE * 127

    def get_arrays_rp(self):
        mat_Rp_obj = np.zeros(self.img.shape)
        mat_Rp_bkg = np.zeros(self.img.shape)
        for edge in self.graph.edges_source: mat_Rp_bkg[edge.head.pos] = edge.residual
        for edge in self.graph.edges_sink: mat_Rp_obj[edge.tail.pos] = edge.residual
        return mat_Rp_obj, mat_Rp_bkg

    def get_arrays_bp(self):
        mat_Bp_top = np.zeros((self.w - 1, self.h))
        mat_Bp_bottom = np.zeros((self.w - 1, self.h))
        mat_Bp_left = np.zeros((self.w, self.h - 1))
        mat_Bp_right = np.zeros((self.w, self.h - 1))

        for node in self.graph.nodes:
            for edge in node.get_neighbor_edges():
                i0,j0 = node.pos
                i1,j1 = edge.head.pos
                if i1 > i0:
                    mat_Bp_bottom[i0, j0] = edge.residual; continue
                if i1 < i0:
                    mat_Bp_top[i0 - 1, j0] = edge.residual; continue
                if j1 > j0:
                    mat_Bp_right[i0, j0] = edge.residual; continue
                if j1 < j0:
                    mat_Bp_left[i0, j0 - 1] = edge.residual

        return mat_Bp_left, mat_Bp_right, mat_Bp_top, mat_Bp_bottom

    def get_array_path_ST(self):
        self.mat_path_ST.fill(0)
        for edge in self.path_ST:
            if edge.tail == self.graph.source: continue
            if edge.head == self.graph.sink: continue
            h0,h1 = edge.head.pos
            t0,t1 = edge.tail.pos
            self.mat_path_ST[h0, h1, 1] = 255
            self.mat_path_ST[h0, h1, 3] = 255
            self.mat_path_ST[t0, t1, 1] = 255
            self.mat_path_ST[t0, t1, 3] = 255

        return self.mat_path_ST

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def start_cut(self): pass
    def continue_cut(self): pass
    def pick_active_node(self): pass
    def remove_active_node(self): pass


# //////////////////////////////////////////////////////////////////////////////
class GraphCutFast(GraphCut):
    def start_cut(self):
        print(">>> Cutting...")
        self.start_time = time.time()

        while True:
            self.grow()
            if not self.path_ST:
                return self.end_cut()
            self.augment()
            self.adopt()
            self.cut_cycle += 1


class GraphCutAnimate(GraphCut):
    def start_cut(self):
        print(">>> Cutting...")
        self.start_time = time.time()

        return self.continue_cut()

    def continue_cut(self):
        while True:
            self.grow()
            if not self.path_ST:
                return self.end_cut()
            self.augment()
            self.adopt()
            self.cut_cycle += 1

            if not self.cut_cycle % CYCLES_PER_FRAME:
                return True


# //////////////////////////////////////////////////////////////////////////////
class GraphCutBFS(GraphCut):
    def pick_active_node(self):
        return self.A_nodes[0]

    def remove_active_node(self):
        self.A_nodes.pop(0)


class GraphCutRND(GraphCut):
    def pick_active_node(self):
        self.random_i = (self.random_i + 1) % len(self.randoms)
        self.r = self.randoms[self.random_i] % len(self.A_nodes)
        return self.A_nodes[self.r]

    def remove_active_node(self):
        self.A_nodes.pop(self.r)


# //////////////////////////////////////////////////////////////////////////////

class GraphCutFastBFS(GraphCutFast, GraphCutBFS): pass
class GraphCutFastRND(GraphCutFast, GraphCutRND): pass
class GraphCutAnimateBFS(GraphCutAnimate, GraphCutBFS): pass
class GraphCutAnimateRND(GraphCutAnimate, GraphCutRND): pass

################################################################################
