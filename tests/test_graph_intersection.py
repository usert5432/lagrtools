import unittest
import numpy as np

from lagrtools.graph import Graph, Edges, Nodes
from lagrtools.intersect import graph_intersection

NODES_DICT1 = {
    'a' : Nodes(
        np.array([ [1], [3], [5], [7], ]),
        np.array([ [2], [4], [6], [8], ]),
    ),
    'b' : Nodes(
        np.array([ [0], [1], [2], ]),
        np.array([ [2], [4], [6], ]),
    ),
}

NODES_DICT2 = {
    'a' : Nodes(
        np.array([ [1], [9], ]), np.array([ [2], [6], ]),
    ),
    'b' : Nodes(
        np.array([ [1], ]), np.array([ [4], ]),
    ),
}

EDGES_DICT1 = {
    ('a', 'b') : Edges(np.array([
        [0, 1],
        [3, 2],
        [2, 0],
    ])),
}

EDGES_DICT2 = {
    ('a', 'b') : Edges(np.array([
        [0, 0],
    ])),
}

class TestsIntersectionFilter(unittest.TestCase):

    def test_simple_intersec(self):
        graph1 = Graph(NODES_DICT1, EDGES_DICT1)
        g_int1, g_int2 = graph_intersection(graph1, graph1)

        self.assertEqual(graph1, g_int1)
        self.assertEqual(graph1, g_int2)

    def test_intersec(self):
        graph1 = Graph(NODES_DICT1, EDGES_DICT1)
        graph2 = Graph(NODES_DICT2, EDGES_DICT2)

        g_int1_null = Graph(
            {
                'a' : Nodes(
                    np.array([ [1], ]), np.array([ [2], ]),
                ),
                'b' : Nodes(
                    np.array([ [1], ]), np.array([ [4], ]),
                ),
            },
            { ('a', 'b') : Edges(np.array([ [0, 0], ])), }
        )

        g_int2_null = Graph(
            {
                'a' : Nodes(
                    np.array([ [1], ]), np.array([ [2], ]),
                ),
                'b' : Nodes(
                    np.array([ [1], ]), np.array([ [4], ]),
                ),
            },
            {
                ('a', 'b') : Edges(np.array([ [0, 0], ])),
            }
        )

        g_int1_test, g_int2_test = graph_intersection(graph1, graph2)

        self.assertEqual(g_int1_test, g_int1_null)
        self.assertEqual(g_int2_test, g_int2_null)


if __name__ == '__main__':
    unittest.main()

