from unittest.mock import patch

import torch
from fastai.data.all import test_eq as _test_eq

from fastaudio.util import create_region_mask, create_sin_wave, test_audio_tensor


def test_create_sin_wave():
    wave = create_sin_wave()
    assert wave != None
    wave, sr = wave
    assert sr == 16000
    assert wave.shape[0] == 5 * sr


def test_shape_of_sin_wave_tensor():
    sr = 16000
    secs = 2
    ai = test_audio_tensor(secs, sr)
    assert ai.duration == secs
    assert ai.nsamples == secs * sr


class TestCreateRegionMask:
    min_size = 4
    max_size = 6

    def test_create_region_mask_max(self):
        # Test max size
        with patch('torch.rand', side_effect=[
                torch.Tensor([[[[1.]]]]),
                torch.Tensor([[[[0.]]]]),
        ]):
            _test_eq(
                create_region_mask([1], self.min_size, self.max_size, 10),
                torch.BoolTensor([[[[1]*6 + [0]*4]]]),
            )

    def test_create_region_mask_min(self):
        # Test min size
        with patch('torch.rand', side_effect=[
                torch.Tensor([0.]),
                # Test start middle start here too
                torch.Tensor([0.5]),
        ]):
            _test_eq(
                create_region_mask([1], self.min_size, self.max_size, 10),
                torch.BoolTensor([0]*3 + [1]*4 + [0]*3),
            )

    def test_create_region_mask_multiple(self):
        # Test multiple masks
        with patch('torch.rand', side_effect=[
                torch.Tensor([[1.], [0. ]]),
                torch.Tensor([[0.], [0.5]]),
        ]):
            _test_eq(
                create_region_mask([2, 1], self.min_size, self.max_size, 10),
                torch.BoolTensor([[1]*6 + [0]*4,
                                  [0]*3 + [1]*4 + [0]*3])
            )
