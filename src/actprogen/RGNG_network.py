"""
RgngGraph
---------
Network Graph used for RGNG (Robust Growing Neural Gas) clustering.
"""
from networkx import Graph


class RgngGraph(Graph):
    ''''Class representing a graph for the RGNG algorithm
    '''
    def add_node(self, node_for_adding, **attr):
        for i in self.nodes():
            self.nodes[i]["prenode_ranking"] += 1
        return super().add_node(node_for_adding, prenode_ranking=0, **attr)
