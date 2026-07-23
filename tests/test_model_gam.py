import unittest

import torch

from model import SACB_Net
from model_gam import GAM_SACB_Net


def make_tiny_model():
    return GAM_SACB_Net(
        inshape=(32, 32, 32),
        ch_scale=2,
        num_k=3,
        token_dim=8,
        token_num_l5=8,
        token_num_l4=8,
        num_types=3,
        fusion_hidden_ch=8,
    )


class GAMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_forward_aux_backward_and_module_gradients(self):
        model = make_tiny_model()
        moving = torch.randn(1, 1, 32, 32, 32)
        fixed = torch.randn(1, 1, 32, 32, 32)
        output = model(moving, fixed, return_aux=True)
        self.assertEqual(output["warped"].shape, moving.shape)
        self.assertEqual(output["flow"].shape, (1, 3, 32, 32, 32))
        self.assertTrue(torch.isfinite(output["warped"]).all())
        self.assertTrue(torch.isfinite(output["flow"]).all())
        required = {
            "context_full",
            "gacm5",
            "gacm4",
            "gate5",
            "gate4",
            "phi5_native",
            "phi4_native",
            "phi3_native",
            "phi2_native",
            "delta4",
        }
        self.assertTrue(required.issubset(output))

        loss = (
            output["warped"].square().mean()
            + 0.01 * output["gacm5"]["cost"].mean()
            + 0.01 * output["gacm4"]["cost"].mean()
        )
        loss.backward()
        parameters = dict(model.named_parameters())
        groups = {
            "gacm5": [value for name, value in parameters.items() if name.startswith("gacm5.")],
            "gacm4": [value for name, value in parameters.items() if name.startswith("gacm4.")],
            "gcdr5": [value for name, value in parameters.items() if name.startswith("gcdr5.")],
            "gcdr4": [value for name, value in parameters.items() if name.startswith("gcdr4.")],
        }
        for name, group in groups.items():
            gradients = [parameter.grad for parameter in group if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), name)
            self.assertGreater(sum(float(gradient.abs().sum().detach()) for gradient in gradients), 0.0, name)
        for parameter_name in (
            "sacb_proj5.cond_proj.weight",
            "sacb_proj4.cond_proj.weight",
            "context_refiner.6.weight",
        ):
            gradient = parameters[parameter_name].grad
            self.assertIsNotNone(gradient, parameter_name)
            self.assertTrue(torch.isfinite(gradient).all(), parameter_name)
            self.assertGreater(float(gradient.abs().sum().detach()), 0.0, parameter_name)

        model.eval()
        with torch.no_grad():
            warped, flow = model(moving, fixed, return_aux=False)
        self.assertEqual(warped.shape, moving.shape)
        self.assertEqual(flow.shape, (1, 3, 32, 32, 32))

    def test_baseline_state_has_no_shared_shape_mismatch(self):
        baseline = SACB_Net(inshape=(32, 32, 32), ch_scale=2, num_k=3)
        model = make_tiny_model()
        baseline_state = baseline.state_dict()
        model_state = model.state_dict()
        shared = sorted(set(baseline_state).intersection(model_state))
        self.assertTrue(shared)
        for key in shared:
            self.assertEqual(baseline_state[key].shape, model_state[key].shape, key)
        result = model.load_state_dict(baseline_state, strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)


if __name__ == "__main__":
    unittest.main()

