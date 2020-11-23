import warnings

import librosa
import torch
from fastai.imports import partial, random
from fastai.vision.augment import RandTransform
from fastcore.meta import delegates
from fastcore.transform import Transform
from fastcore.utils import ifnone
from torch.nn import functional as F
from torch import Tensor

from ..core.spectrogram import AudioSpectrogram, AudioTensor
from ..util import create_region_mask, auto_batch
from .signal import AudioPadType


class SpectrogramTransform(Transform):
    "Helps prevent us trying to apply to Audio Tensors"

    def encodes(self, audio: AudioTensor) -> AudioTensor:
        warnings.warn(f"You are trying to apply a {type(self).__name__} to an AudioTensor {audio.shape}")
        return audio


class CropTime(SpectrogramTransform):
    """Random crops full spectrogram to be length specified in ms by crop_duration"""

    def __init__(self, duration, pad_mode=AudioPadType.Zeros):
        self.duration = duration
        self.pad_mode = pad_mode

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        sr, hop = sg.sr, sg.hop_length
        w_crop = int((sr * self.duration) / (1000 * hop)) + 1
        w_sg = sg.shape[-1]
        if w_sg == w_crop:
            sg_crop = sg
        elif w_sg < w_crop:
            sg_crop = _tfm_pad_spectro(sg, w_crop, pad_mode=self.pad_mode)
        else:
            crop_start = random.randint(0, int(w_sg - w_crop))
            sg_crop = sg[:, :, crop_start : crop_start + w_crop]
            sg_crop.sample_start = int(crop_start * hop)
            sg_crop.sample_end = sg_crop.sample_start + int(self.duration * sr)
        sg.data = sg_crop
        return sg


def _tfm_pad_spectro(sg, width, pad_mode=AudioPadType.Zeros):
    """Pad spectrogram to specified width, using specified pad mode"""
    c, y, x = sg.shape
    if pad_mode in [AudioPadType.Zeros, AudioPadType.Zeros_After]:
        padded = torch.zeros((c, y, width))
        start = random.randint(0, width - x) if pad_mode == AudioPadType.Zeros else 0
        padded[:, :, start : start + x] = sg.data
        return padded
    elif pad_mode == AudioPadType.Repeat:
        repeats = width // x + 1
        return sg.repeat(1, 1, repeats)[:, :, :width]
    else:
        raise ValueError(
            f"""pad_mode {pad_mode} not currently supported,
            only AudioPadType.Zeros, AudioPadType.Zeros_After,
            or AudioPadType.Repeat"""
        )


def mask_along_axis(specgrams, num_masks=1, min_size=1, max_size=10, mask_val=None, axis=2):
    device = specgrams.device
    shape = specgrams.size()
    # TODO: Still need this?
    dtype = specgrams.dtype

    # First create the broadcastable masks. Each spectrogram gets its own set of
    # masks.
    #
    # The masks are created in parallel by adding an extra dimension to the mask
    # shape, then collapsing it down to combine them.
    masks = create_region_mask(
        [num_masks, shape[0], 1, 1, 1], min_size, max_size, shape[axis], device=device
    ).amax(dim=0)

    # Orient so the axis we're masking comes last
    specgrams = specgrams.transpose(axis, -1)
    # Now mask it
    if mask_val:
        specgrams.masked_fill_(masks, mask_val)
    else:
        # Mask with the channel-wise mean. Note while each channel in a given
        # spectrogram takes the same mask position, the value is determined
        # per-channel.

        # Take the mean of the masked area, not the whole spectrogram, so as not
        # to change the overall mean (although this will affect the standard
        # deviation).
        mask_vals = (specgrams.mul(masks).sum((-2, -1))
                     # This will be broadcast, so we have to manually multiply
                     # by the size of that dimension.
                     / (masks.sum((-2, -1)) * specgrams.shape[-2]))

        # Alternate method: whole channel mean. Use `reshape` because it
        # might not be contiguous.
        # mask_vals = specgrams.reshape(*specgrams.shape[:2], -1).mean(-1)

        # TODO: Can this be done inplace?
        specgrams = torch.where(masks, mask_vals[..., None, None], specgrams)
        # TODO: This can't broadcast the values, unfortunately.
        # specgrams.masked_scatter_(combined_masks, mask_vals[..., None, None])
    # Restore original orientation
    return specgrams.transpose(axis, -1)


class _MaskAxis(RandTransform):
    """Base class for SpecAugment masking transforms."""
    def __init__(self, axis, num_masks=1, min_size=1, max_size=10, mask_val=None, batch=False):
        if axis not in [2, 3]:
            raise ValueError("Can only mask the time or frequency axis (2 or 3)")
        self.num_masks = num_masks
        self.min_size  = min_size
        self.max_size  = max_size
        self.mask_val  = mask_val
        self.axis      = axis
        self.batch     = batch
        super().__init__()

    @auto_batch(3)
    def encodes(self, sg: AudioSpectrogram):
        return mask_along_axis(sg, num_masks=self.num_masks,
                               min_size=self.min_size, max_size=self.max_size,
                               mask_val=self.mask_val, axis=self.axis)


@delegates(_MaskAxis)
class MaskFreq(_MaskAxis):
    """Google SpecAugment frequency masking from https://arxiv.org/abs/1904.08779.

    This version runs on batches and can be run efficiently on the GPU.

    """
    # TODO: How to delegate the kwargs
    def __init__(self, **kwargs):
        super().__init__(axis=2, **kwargs)


@delegates(_MaskAxis)
class MaskTime(_MaskAxis):
    """Google SpecAugment time masking from https://arxiv.org/abs/1904.08779.

    This version runs on batches and can be run efficiently on the GPU.

    """
    # TODO: How to delegate the kwargs
    def __init__(self, **kwargs):
        super().__init__(axis=3, **kwargs)


class SGRoll(SpectrogramTransform):
    """Shifts spectrogram along x-axis wrapping around to other side"""

    def __init__(self, max_shift_pct=0.5, direction=0):
        if int(direction) not in [-1, 0, 1]:
            raise ValueError("Direction must be -1(left) 0(bidirectional) or 1(right)")
        self.max_shift_pct = max_shift_pct
        self.direction = direction

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        direction = random.choice([-1, 1]) if self.direction == 0 else self.direction
        w = sg.shape[-1]
        roll_by = int(w * random.random() * self.max_shift_pct * direction)
        sg.data = sg.roll(roll_by, dims=-1)
        return sg


def _torchdelta(sg: AudioSpectrogram, order=1, width=9):
    """Converts to numpy, takes delta and converts back to torch, needs torchification"""
    if sg.shape[1] < width:
        raise ValueError(
            f"""Delta not possible with current settings, inputs must be wider than
        {width} columns, try setting max_to_pad to a larger value to ensure a minimum width"""
        )
    return AudioSpectrogram(
        torch.from_numpy(librosa.feature.delta(sg.numpy(), order=order, width=width))
    )


class Delta(SpectrogramTransform):
    """Creates delta with order 1 and 2 from spectrogram
    and concatenate with the original"""

    def __init__(self, width=9):
        self.td = partial(_torchdelta, width=width)

    def encodes(self, sg: AudioSpectrogram):
        new_channels = [
            torch.stack([c, self.td(c, order=1), self.td(c, order=2)]) for c in sg
        ]
        sg.data = torch.cat(new_channels, dim=0)
        return sg


class TfmResize(SpectrogramTransform):
    """Temporary fix to allow image resizing transform"""

    def __init__(self, size, interp_mode="bilinear"):
        self.size = size
        self.interp_mode = interp_mode

    def encodes(self, sg: AudioSpectrogram) -> AudioSpectrogram:
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        c, y, x = sg.shape
        sg.data = F.interpolate(
            sg.unsqueeze(0), size=self.size, mode=self.interp_mode, align_corners=False
        ).squeeze(0)
        return sg
