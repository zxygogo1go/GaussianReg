import unittest

import torch

from utils import SpatialTransformer


class SpatialTransformerTests(unittest.TestCase):
    def test_cpu_identity_and_dhw_flow_direction(self):
        transformer = SpatialTransformer((4, 5, 6))
        source = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(1, 1, 4, 5, 6)
        identity = transformer(source, torch.zeros(1, 3, 4, 5, 6))
        self.assertTrue(torch.allclose(identity, source, atol=1.0e-5))

        flow = torch.zeros(1, 3, 4, 5, 6)
        flow[:, 0] = 1.0
        shifted = transformer(source, flow)
        self.assertTrue(torch.allclose(shifted[:, :, :-1], source[:, :, 1:], atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
