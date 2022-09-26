import unittest
import numpy as np

from lagrtools.graph import Graph, Edges, Nodes

NODES_DICT = {
    'a' : Nodes(
        np.array([ [1], [3], [5], [7], ]),
        np.array([ [2], [4], [6], [8], ]),
    ),
    'b' : Nodes(
        np.array([ [0], [1], [2], ]),
        np.array([ [2], [4], [6], ]),
    ),
}

EDGES_DICT = {
    ('a', 'b') : Edges(np.array([
        [0, 1],
        [3, 2],
    ])),
}

class TestsGraphFilter(unittest.TestCase):

    def test_simple_eq(self):
        graph1 = Graph(NODES_DICT, EDGES_DICT)
        graph2 = Graph(NODES_DICT, EDGES_DICT)

        self.assertEqual(graph1, graph2)

    def test_filter_simple(self):
        graph = Graph(NODES_DICT, EDGES_DICT)
        mask_dict = {}

        self.assertEqual(graph.filter(mask_dict), graph)

    def test_filter_1(self):
        mask_dict  = { 'a' : [ True, True, False, False ], }
        nodes_dict = {
            'a' : Nodes(
                np.array([ [1], [3], ]), np.array([ [2], [4], ]),
            ),
            'b' : Nodes(
                np.array([ [0], [1], [2], ]), np.array([ [2], [4], [6], ]),
            ),
        }
        edges_dict = { ('a', 'b') : Edges(np.array([ [0, 1], ])), }

        graph1 = Graph(NODES_DICT, EDGES_DICT)
        graph2 = Graph(nodes_dict, edges_dict)

        self.assertEqual(graph1.filter(mask_dict), graph2)

    def test_filter_2(self):
        mask_dict  = { 'b' : [ True, True, False ], }
        nodes_dict = {
            'a' : Nodes(
                np.array([ [1], [3], [5], [7], ]),
                np.array([ [2], [4], [6], [8], ]),
            ),
            'b' : Nodes(
                np.array([ [0], [1], ]), np.array([ [2], [4], ]),
            ),
        }
        edges_dict = {
            ('a', 'b') : Edges(np.array([ [0, 1], ])),
        }

        graph1 = Graph(NODES_DICT, EDGES_DICT)
        graph2 = Graph(nodes_dict, edges_dict)

        self.assertEqual(graph1.filter(mask_dict), graph2)

    def test_filter_3(self):
        mask_dict  = {
            'a' : [ False, True, False, False ],
            'b' : [ True, True, False ],
        }
        nodes_dict = {
            'a' : Nodes( np.array([ [3], ]), np.array([ [4], ]),),
            'b' : Nodes( np.array([ [0], [1], ]), np.array([ [2], [4], ]),),
        }
        edges_dict = { ('a', 'b') : Edges(np.zeros((0, 2))), }

        graph1 = Graph(NODES_DICT, EDGES_DICT)
        graph2 = Graph(nodes_dict, edges_dict)

        self.assertEqual(graph1.filter(mask_dict), graph2)


if __name__ == '__main__':
    unittest.main()

