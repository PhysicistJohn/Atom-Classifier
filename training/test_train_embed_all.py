from __future__ import annotations

import unittest
import warnings

import numpy as np
import torch

from train import embed_all


class _Probe(torch.nn.Module):
    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (x.mean(dim=(1, 2), keepdim=False).unsqueeze(1), feat[:, :1]),
            dim=1,
        )


class EmbedAllTest(unittest.TestCase):
    def test_read_only_cached_arrays_do_not_alias_torch_inputs(self):
        x = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
        feat = np.arange(12, dtype=np.float32).reshape(4, 3)
        x.setflags(write=False)
        feat.setflags(write=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = embed_all(_Probe(), x, feat, torch.device("cpu"), batch=2)

        expected = np.stack((x.mean(axis=(1, 2)), feat[:, 0]), axis=1)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
