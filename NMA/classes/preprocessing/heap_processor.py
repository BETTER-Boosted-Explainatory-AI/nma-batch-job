import heapq
import copy

from utilss.enums.heap_types import HeapType
from utilss.enums.graph_types import GraphTypes

class HeapProcessor:
    def __init__(self,graph, graph_type, labels):
        self.heap_type = self._get_heap_type(graph_type)
        self.heap = []
        self.nodes_multiplier = -1 if self.heap_type == "max" else 1
        
        self._process_edges(graph, labels)
 
        
    def _get_heap_type(self, graph_type):
        return HeapType.MINIMUM.value if graph_type == GraphTypes.DISSIMILARITY.value else HeapType.MAXIMUM.value

    def _process_edges(self, graph, labels):
        """Processes edges and pushes them into a max heap."""
        for edge in graph.es:
            source = graph.vs[edge.source]["name"]
            target = graph.vs[edge.target]["name"]
            weight = edge["weight"] if "weight" in edge.attributes() else 0
            heapq.heappush(self.heap, (self.nodes_multiplier * weight, source, target))

    def get_heap(self):
        """Returns the max heap."""
        return self.heap

    def get_heap_copy(self):
        """Returns a deep copy of the max heap."""
        return copy.deepcopy(self.heap)
