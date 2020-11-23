from enum import Enum

import torch
import torch.fft
from fastai.imports import np, random
from fastai.vision.augment import RandTransform
from fastcore.transform import Transform

from fastaudio.util import colored_noise, create_region_mask, auto_batch

from ..core.signal import AudioTensor
from ..core.spectrogram import AudioSpectrogram


class AudioPadType(Enum):
    "All methods of padding audio as attributes to get tab-completion and typo-proofing",
    Zeros = "zeros"
    Zeros_After = "zeros_after"
    Repeat = "repeat"


class ResizeSignal(Transform):
    """Crops signal to be length specified in ms by duration, padding if needed"""

    def __init__(self, duration, pad_mode=AudioPadType.Zeros):
        self.duration = duration
        self.pad_mode = pad_mode
        if pad_mode not in [
            AudioPadType.Zeros,
            AudioPadType.Zeros_After,
            AudioPadType.Repeat,
        ]:
            raise ValueError(
                f"""pad_mode {pad_mode} not currently supported,
                only AudioPadType.Zeros, AudioPadType.Zeros_After,
                or AudioPadType.Repeat"""
            )

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        sig = ai.data
        orig_samples = ai.nsamples
        crop_samples = int((self.duration / 1000) * ai.sr)
        if orig_samples == crop_samples:
            return ai
        elif orig_samples < crop_samples:
            ai.data = _tfm_pad_signal(sig, crop_samples, pad_mode=self.pad_mode)
        else:
            crop_start = random.randint(0, int(orig_samples - crop_samples))
            ai.data = sig[:, crop_start : crop_start + crop_samples]
        return ai


def _tfm_pad_signal(sig, width, pad_mode=AudioPadType.Zeros):
    """Pad spectrogram to specified width, using specified pad mode"""
    c, x = sig.shape
    if pad_mode in [AudioPadType.Zeros, AudioPadType.Zeros_After]:
        zeros_front = (
            random.randint(0, width - x) if pad_mode == AudioPadType.Zeros else 0
        )
        pad_front = torch.zeros((c, zeros_front))
        pad_back = torch.zeros((c, width - x - zeros_front))
        return torch.cat((pad_front, sig, pad_back), 1)
    elif pad_mode == AudioPadType.Repeat:
        repeats = width // x + 1
        return sig.repeat(1, repeats)[:, :width]


def _shift(sig, s):
    if s == 0:
        return sig
    out = torch.zeros_like(sig)
    if s < 0:
        out[..., :s] = sig[..., -s:]
    else:
        out[..., s:] = sig[..., :-s]
    return out


def shift_signal(t: torch.Tensor, shift, roll):
    # refactor 2nd half of this statement to just take and roll the final axis
    if roll:
        t.data = torch.from_numpy(np.roll(t.numpy(), shift, axis=-1))
    else:
        t.data = _shift(t, shift)
    return t


class SignalShifter(RandTransform):
    """Randomly shifts the audio signal by `max_pct` %.
    direction must be -1(left) 0(bidirectional) or 1(right).
    """

    def __init__(self, p=0.5, max_pct=0.2, max_time=None, direction=0, roll=False):
        if direction not in [-1, 0, 1]:
            raise ValueError("Direction must be -1(left) 0(bidirectional) or 1(right)")
        self.max_pct = max_pct
        self.max_time = max_time
        self.direction = direction
        self.roll = roll
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.shift_factor = random.uniform(-1, 1)
        if self.direction != 0:
            self.shift_factor = self.direction * abs(self.shift_factor)

    def encodes(self, ai: AudioTensor):
        if self.max_time is None:
            s = self.shift_factor * self.max_pct * ai.nsamples
        else:
            s = self.shift_factor * self.max_time * ai.sr
        ai.data = shift_signal(ai.data, int(s), self.roll)
        return ai

    def encodes(self, sg: AudioSpectrogram):
        if self.max_time is None:
            s = self.shift_factor * self.max_pct * sg.width
        else:
            s = self.shift_factor * self.max_time * sg.sr
        return shift_signal(sg, int(s), self.roll)


class NoiseColor:
    Violet = -2
    Blue = -1
    White = 0
    Pink = 1
    Brown = 2


class AddNoise(RandTransform):
    "Adds noise of specified color and level to the audio signal"

    def __init__(self, p=1., min_level=0.0, max_level=0.05, color=NoiseColor.White):
        self.min_level = min_level
        self.max_level = max_level
        self.color     = color
        if color not in [*range(-2, 3)]:
            raise ValueError(f"color {color} is not valid")
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        self.do = True

    @auto_batch(2)
    def encodes(self, ai: AudioTensor) -> AudioTensor:
        n = ai.shape[0]

        if self.color == NoiseColor.White:
            # if it's white noise, implement our own for speed
            noise = torch.randn_like(ai.data)
        else:
            noise = colored_noise(ai.shape, exponent=self.color, device=ai.device)

        # Noise is *not* calcualted per-channel, but per-item.
        noise *= (ai.abs().mean((-2, -1))
                  # Scales
                  * (torch.randn([n], device=ai.device)
                     * (self.max_level - self.min_level) + self.min_level)
                  # Mask
                  * (torch.randn([n], device=ai.device) <= self.p)
                 )[..., None, None]
        ai.data += noise
        return ai


class ChangeVolume(RandTransform):
    "Changes the volume of the signal"

    def __init__(self, p=0.5, lower=0.5, upper=1.5):
        self.lower, self.upper = lower, upper
        self._rand_mult = upper - lower
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        self.do = True

    @auto_batch(2)
    def encodes(self, ai: AudioTensor) -> AudioTensor:
        op_shape = [ai.size(0)]
        scales = torch.where(
            # Mask
            torch.rand(op_shape, device=ai.device) <= self.p,
            # Modifier
            torch.rand(op_shape, device=ai.device) * self._rand_mult + self.lower,
            # No modification
            torch.ones(op_shape, device=ai.device),
        )
        ai *= scales[:, None, None]
        return ai


class SignalCutout(RandTransform):
    "Randomly zeros some portion of the signal"

    def __init__(self, p=0.5, max_cut_pct=0.15):
        self.max_cut_pct = max_cut_pct
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        self.do = True

    @auto_batch(2)
    def encodes(self, ai: AudioTensor):
        n, c, s = ai.shape

        mask = create_region_mask([n, 1, 1], 0, (s * self.max_cut_pct), s, device=ai.device)
        # Only apply mask to a random subset of items
        mask *= torch.rand([n, 1, 1], device=ai.device) <= self.p
        ai.masked_fill_(mask, 0)

        return ai


class SignalLoss(RandTransform):
    "Randomly loses some portion of samples"

    def __init__(self, p=0.5, max_cut_pct=0.15):
        self.max_cut_pct = max_cut_pct
        super().__init__(p=p)

    def before_call(self, b, split_idx):
        self.do = True

    @auto_batch(2)
    def encodes(self, ai: AudioTensor):
        n, c, s = ai.shape

        # TODO: Lose the same points across channels?
        cut_pcts = torch.rand([n, 1, 1], device=ai.device) * self.max_cut_pct
        masks = torch.rand([n, c, s], device=ai.device) <= cut_pcts

        # Only use a certain proportion of the masks
        masks *= torch.rand([n, 1, 1], device=ai.device) <= self.p

        ai.masked_fill_(masks, 0)

        return ai


# downmixMono was removed from torchaudio, we now just take the mean across channels
# this works for both batches and individual items
class DownmixMono(Transform):
    "Transform multichannel audios into single channel"

    def encodes(self, ai: AudioTensor) -> AudioTensor:
        downmixed = ai.data.contiguous().mean(-2).unsqueeze(-2)
        return AudioTensor(downmixed, ai.sr)
