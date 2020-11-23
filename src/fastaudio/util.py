from math import pi
from functools import wraps

import torch
from torch import Tensor
from fastai.vision.augment import RandTransform

from fastaudio.core.signal import AudioTensor

__all__ = ["test_audio_tensor"]


def create_sin_wave(seconds=5, sr=16000, freq=400):
    "Creates a sin wave to be used for tests"
    max_samples = freq * seconds * 2 * pi
    rate = 2 * pi * freq / sr
    samples = torch.arange(0, max_samples, rate, dtype=torch.float)
    sin_wave = torch.sin(samples)
    return sin_wave, sr


def test_audio_tensor(seconds=2, sr=16000, channels=1):
    "Generates an Audio Tensor for testing that is based on a sine wave"
    sin_wave, sr = create_sin_wave(seconds=seconds, sr=sr)
    sin_wave = sin_wave.repeat(channels, 1)
    return AudioTensor(sin_wave, sr)


def apply_transform(transform, inp):
    """Generate a new input, apply transform, and display/return both input and output"""
    inp_orig = inp.clone()
    out = (
        transform(inp_orig, split_idx=0)
        if isinstance(transform, RandTransform)
        else transform(inp_orig)
    )
    return inp.clone(), out


def create_region_mask(shape, min_mask_size, max_mask_size, maskable_area, device=None):
    # Generate the start & end positions for each mask, then compare these to absolute
    # indices to create the masks.
    mask_sizes  = torch.rand(shape, device=device) * (max_mask_size - min_mask_size) + min_mask_size
    mask_starts = torch.rand(shape, device=device) * (maskable_area - mask_sizes)
    mask_ends = mask_starts + mask_sizes
    indexes = torch.arange(0, maskable_area, device=device)
    return (mask_starts <= indexes) & (indexes < mask_ends)


def _rfftfreq(n, d=1.0, device=None):
    """Get the sample frequencies for a discrete fourier transform."""
    # TODO: Document properly
    val = 1.0/(n * d)
    results = torch.arange(0, n//2 + 1, device=device, dtype=int)
    return results * val


def colored_noise(shape, exponent, fmin=0, device=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance

    Ported to PyTorch from Felix Patzelt's numpy implementation.
    https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py#L9

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.

    Returns
    -------

    out : array
        The samples.

    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> y = colored_noise([1], 5)

    """
    # The number of samples in each time series
    nsamples = shape[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    # FIXME: torch.fft is missing this for some reason
    # f = torch.fft.rfftfreq(nsamples)
    f = _rfftfreq(nsamples, device=device)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./nsamples) # Low frequency cutoff
    ix   = (s_scale < fmin).sum()   # Index of the cutoff
    # TODO: What if it's bigger though
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].clone()
    w[-1] *= (1 + (nsamples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * (w**2).sum().sqrt() / nsamples

    # Adjust size to generate one Fourier component per frequency
    #
    # TODO: This seems quite wrong. Won't this return a different shape?
    new_shape = (*shape[:-1], f.size(0))
    # Original numpy imp
    # shape[-1] = f.size(0)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(new_shape) - 1
    s_scale     = s_scale[[None] * dims_to_add + [Ellipsis]]

    # Generate scaled random power + phase
    sr = torch.randn(new_shape, device=device) * s_scale
    si = torch.randn(new_shape, device=device) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (nsamples % 2): si[...,-1] = 0

    # Regardless of signal length, the DC component must be real
    si[...,0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = torch.fft.irfft(s, n=nsamples, dim=-1) / sigma

    return y


def auto_batch(item_dims):
    """Wrapper that always calls the underlying function with a batch.

    Single items are expanded before calling, then the result is returned with
    their original shape.

    """
    def wrapper(orig_func):
        @wraps(orig_func)
        def expand_and_do(self, x: Tensor):
            nonlocal orig_func, item_dims
            # Expand to batch
            shape = x.size()
            x = x.reshape(-1, *shape[-item_dims:])

            x = orig_func(self, x)

            # Restore original shape
            return x.reshape(*shape)

        return expand_and_do
    return wrapper
