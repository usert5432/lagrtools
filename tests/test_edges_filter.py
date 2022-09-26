import unittest
import numpy as np

from lagrtools.edges import Edges

class TestsEdgeFilter(unittest.TestCase):

    def test_simple_eq(self):
        adj_list = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        edges1 = Edges(adj_list)
        edges2 = Edges(adj_list)

        self.assertEqual(edges1, edges2)

    def test_filter_1(self):
        edges1 = Edges(np.array([
            [0, 1],
            [0, 2],
            [0, 3],
        ]))
        edges2 = edges1.filter(None, None)

        self.assertEqual(edges1, edges2)

    def test_filter_2(self):
        edges1 = Edges(np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ]))
        edges2 = Edges(np.array([
            [1, 0],
            [3, 1],
        ]))

        mask_src = None
        mask_dst = np.array([ False, True, False, True ])

        self.assertEqual(edges1.filter(mask_src, mask_dst), edges2)

    def test_filter_3(self):
        edges1 = Edges(np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ]))
        edges2 = Edges(np.array([
            [0, 1],
            [1, 3],
        ]))

        mask_src = np.array([ False, True, False, True ])
        mask_dst = None

        self.assertEqual(edges1.filter(mask_src, mask_dst), edges2)

    def test_filter_4(self):
        edges1 = Edges(np.array([
            [4, 0],
            [3, 1],
            [2, 2],
            [1, 3],
            [0, 4],
        ]))
        edges2 = Edges(np.array([
            [2, 0],
            [1, 1],
            [0, 2],
        ]))

        mask_src = np.array([ True,  True, False, True, False ])
        mask_dst = np.array([ False, True, False, True, True  ])

        self.assertEqual(edges1.filter(mask_src, mask_dst), edges2)

if __name__ == '__main__':
    unittest.main()

