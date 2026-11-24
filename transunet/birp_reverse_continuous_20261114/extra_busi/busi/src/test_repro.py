"""Analytic/numerical resize checks, RNG isolation, sampler and loss contracts."""
import os
import random
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from repro_utils import (DeterministicBilinear, _Bilinear, deterministic_resize_modules,
    digest, pixel_cross_entropy, seed_worker, source_indices, stream_seed, target_indices)


class ReproTests(unittest.TestCase):
    def test_ema_all_uses_teacher_labels_and_all_pixels(self):
        from train_repro import build_reliable_labels
        class Teacher(nn.Module):
            def forward(self, x):
                return torch.tensor([[[[4., -4.]], [[-4., 4.]]]])
        pseudo, mask, stats = build_reliable_labels(
            Teacher(), None, torch.zeros(1, 1, 1, 2), 'ema_all',
            .9, .85, .12, .6, .55, .25, .02, 1.5, .05)
        self.assertTrue(torch.equal(pseudo, torch.tensor([[[0, 1]]])))
        self.assertTrue(torch.equal(mask, torch.ones(1, 1, 2)))
        self.assertEqual(stats['pseudo_cov'], 1.0)

    def test_bilinear_forward_exact(self):
        for align in (False, True):
            for shape, size in (((2, 3, 3, 5), (6, 10)), ((1, 2, 7, 9), (4, 6)), ((1, 1, 1, 2), (3, 4))):
                x = torch.randn(shape, dtype=torch.float64)
                a = _Bilinear.apply(x, size, align)
                b = F.interpolate(x, size=size, mode="bilinear", align_corners=align)
                self.assertTrue(torch.equal(a, b))

    def test_bilinear_gradient_matches_native(self):
        for align in (False, True):
            for size in ((6, 10), (2, 3), (1, 1)):
                x = torch.randn(2, 3, 3, 5, dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                grad = torch.randn(2, 3, *size, dtype=torch.float64)
                _Bilinear.apply(x, size, align).backward(grad)
                F.interpolate(y, size=size, mode="bilinear", align_corners=align).backward(grad)
                self.assertTrue(torch.allclose(x.grad, y.grad, atol=1e-12, rtol=1e-12))

    def test_gradcheck(self):
        x = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        for align in (False, True):
            self.assertTrue(torch.autograd.gradcheck(lambda a: _Bilinear.apply(a, (4, 6), align), (x,)))

    def test_replacement_keeps_state_names_and_forward(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.UpsamplingBilinear2d(scale_factor=2))
        x = torch.randn(1, 1, 4, 4)
        before, keys = model(x), list(model.state_dict())
        self.assertEqual(deterministic_resize_modules(model), ["1"])
        self.assertEqual(keys, list(model.state_dict()))
        self.assertTrue(torch.equal(before, model(x)))

    def test_ce_value_and_gradient(self):
        x = torch.randn(2, 2, 5, 7, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        labels = torch.randint(0, 2, (2, 5, 7))
        a, b = pixel_cross_entropy(x, labels), F.cross_entropy(y, labels, reduction="none")
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        a.mean().backward()
        b.mean().backward()
        self.assertTrue(torch.allclose(x.grad, y.grad, atol=1e-7))

    def test_sampler_independent_of_global_rng_and_gamma(self):
        s = source_indices(11, 2, 1337, 3)
        t = target_indices(13, 2, 5, 1337, 3)
        before = digest(torch.get_rng_state())
        self.assertEqual(s, source_indices(11, 2, 1337, 3))
        self.assertEqual(t, target_indices(13, 2, 5, 1337, 3))
        self.assertEqual(before, digest(torch.get_rng_state()))
        torch.rand(100)
        self.assertEqual(s, source_indices(11, 2, 1337, 3))
        self.assertEqual(t, target_indices(13, 2, 5, 1337, 3))
        self.assertEqual(len(s), len(set(s)))

    def test_target_cycle_resume(self):
        first = target_indices(7, 2, 5, 2027, 1)
        second = target_indices(7, 2, 5, 2027, 2)
        combined = target_indices(7, 2, 10, 2027, 1)
        self.assertEqual(first + second, combined)
        self.assertEqual(len(combined), 20)
        self.assertTrue(all(0 <= i < 7 for i in combined))

    def test_worker_stream_separate_from_model_stream(self):
        data = torch.utils.data.TensorDataset(torch.arange(10))
        before = digest(torch.get_rng_state())
        loader = torch.utils.data.DataLoader(data, batch_size=2,
            generator=torch.Generator().manual_seed(stream_seed(1337, "validation", 0)))
        list(loader)
        self.assertEqual(before, digest(torch.get_rng_state()))

    def test_rng_and_state_hashes_include_numpy_and_scalar_tensors(self):
        self.assertEqual(digest(torch.tensor(1)), digest(torch.tensor(1)))
        self.assertNotEqual(digest(torch.tensor(1)), digest(torch.tensor(2)))
        self.assertEqual(digest(np.random.get_state()), digest(np.random.get_state()))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU contract")
    def test_gpu_bilinear_repeat_exact_under_strict_policy(self):
        from repro_utils import configure_strict
        configure_strict(1337)
        x = torch.randn(2, 3, 14, 14, device="cuda", requires_grad=True)
        grad = torch.randn(2, 3, 28, 28, device="cuda")
        native = F.interpolate(x.detach(), size=(28, 28), mode="bilinear", align_corners=True)
        a = _Bilinear.apply(x, (28, 28), True)
        self.assertTrue(torch.equal(a, native))
        a.backward(grad)
        one = x.grad.detach().clone()
        x.grad = None
        _Bilinear.apply(x, (28, 28), True).backward(grad)
        self.assertTrue(torch.equal(one, x.grad))


if __name__ == "__main__":
    unittest.main()
