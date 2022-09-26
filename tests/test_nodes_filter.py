import unittest
import numpy as np

from lagrtools.nodes import Nodes

class TestsNodeFilter(unittest.TestCase):

    def test_simple_eq(self):
        nodes1 = Nodes(np.array([1,2,3])[:, np.newaxis], np.array([1,2,3]),)
        nodes2 = Nodes(np.array([1,2,3])[:, np.newaxis], np.array([1,2,3]),)

        self.assertEqual(nodes1, nodes2)

    def test_filter_1(self):
        nodes1 = Nodes(np.array([1,2,3])[:, np.newaxis], np.array([1,2,3]),)
        nodes2 = Nodes(np.array([1,3])[:, np.newaxis], np.array([1,3]),)

        mask = np.array([True, False, True])
        self.assertEqual(nodes1.filter(mask), nodes2)

    def test_filter_2(self):
        nodes1 = Nodes(
            np.array([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ]),
            np.array([ [2], [4], [6], [8], ]),
        )
        nodes2 = Nodes(
            np.array([
                [5, 6],
            ]),
            np.array([ [6], ]),
        )

        mask = np.array([False, False, True, False])
        self.assertEqual(nodes1.filter(mask), nodes2)

    def test_filter_3(self):
        nodes1 = Nodes(
            np.array([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ]),
            np.array([ [2], [4], [6], [8], ]),
        )
        nodes2 = Nodes(
            np.array([
                [1, 2],
                [3, 4],
                [7, 8],
            ]),
            np.array([ [2], [4], [8], ]),
        )

        mask = np.array([True, True, False, True])
        self.assertEqual(nodes1.filter(mask), nodes2)

if __name__ == '__main__':
    unittest.main()
