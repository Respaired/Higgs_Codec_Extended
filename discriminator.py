# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from audiotools import AudioSignal
# from audiotools import ml
# from audiotools import STFTParams
# from einops import rearrange
# from torch.nn.utils import weight_norm


# def WNConv1d(*args, **kwargs):
#     act = kwargs.pop("act", True)
#     conv = weight_norm(nn.Conv1d(*args, **kwargs))
#     if not act:
#         return conv
#     return nn.Sequential(conv, nn.LeakyReLU(0.1))


# def WNConv2d(*args, **kwargs):
#     act = kwargs.pop("act", True)
#     conv = weight_norm(nn.Conv2d(*args, **kwargs))
#     if not act:
#         return conv
#     return nn.Sequential(conv, nn.LeakyReLU(0.1))


# class MPD(nn.Module):
#     def __init__(self, period):
#         super().__init__()
#         self.period = period
#         self.convs = nn.ModuleList(
#             [
#                 WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
#                 WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
#                 WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
#                 WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
#                 WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
#             ]
#         )
#         self.conv_post = WNConv2d(
#             1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
#         )

#     def pad_to_period(self, x):
#         t = x.shape[-1]
#         x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
#         return x

#     def forward(self, x):
#         fmap = []

#         x = self.pad_to_period(x)
#         x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

#         for layer in self.convs:
#             x = layer(x)
#             fmap.append(x)

#         x = self.conv_post(x)
#         fmap.append(x)

#         return fmap


# class MSD(nn.Module):
#     def __init__(self, rate: int = 1, sample_rate: int = 44100):
#         super().__init__()
#         self.convs = nn.ModuleList(
#             [
#                 WNConv1d(1, 16, 15, 1, padding=7),
#                 WNConv1d(16, 64, 41, 4, groups=4, padding=20),
#                 WNConv1d(64, 256, 41, 4, groups=16, padding=20),
#                 WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
#                 WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
#                 WNConv1d(1024, 1024, 5, 1, padding=2),
#             ]
#         )
#         self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
#         self.sample_rate = sample_rate
#         self.rate = rate

#     def forward(self, x):
#         x = AudioSignal(x, self.sample_rate)
#         x.resample(self.sample_rate // self.rate)
#         x = x.audio_data

#         fmap = []

#         for l in self.convs:
#             x = l(x)
#             fmap.append(x)
#         x = self.conv_post(x)
#         fmap.append(x)

#         return fmap


# BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


# class MRD(nn.Module):
#     def __init__(
#         self,
#         window_length: int,
#         hop_factor: float = 0.25,
#         sample_rate: int = 44100,
#         bands: list = BANDS,
#     ):
#         """Complex multi-band spectrogram discriminator.
#         Parameters
#         ----------
#         window_length : int
#             Window length of STFT.
#         hop_factor : float, optional
#             Hop factor of the STFT, defaults to ``0.25 * window_length``.
#         sample_rate : int, optional
#             Sampling rate of audio in Hz, by default 44100
#         bands : list, optional
#             Bands to run discriminator over.
#         """
#         super().__init__()

#         self.window_length = window_length
#         self.hop_factor = hop_factor
#         self.sample_rate = sample_rate
#         self.stft_params = STFTParams(
#             window_length=window_length,
#             hop_length=int(window_length * hop_factor),
#             match_stride=True,
#         )

#         n_fft = window_length // 2 + 1
#         bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
#         self.bands = bands

#         ch = 32
#         convs = lambda: nn.ModuleList(
#             [
#                 WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
#                 WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
#             ]
#         )
#         self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
#         self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

#     def spectrogram(self, x):
#         x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
#         x = torch.view_as_real(x.stft())
#         x = rearrange(x, "b 1 f t c -> (b 1) c t f")
#         # Split into bands
#         x_bands = [x[..., b[0] : b[1]] for b in self.bands]
#         return x_bands

#     def forward(self, x):
#         x_bands = self.spectrogram(x)
#         fmap = []

#         x = []
#         for band, stack in zip(x_bands, self.band_convs):
#             for layer in stack:
#                 band = layer(band)
#                 fmap.append(band)
#             x.append(band)

#         x = torch.cat(x, dim=-1)
#         x = self.conv_post(x)
#         fmap.append(x)

#         return fmap


# class Discriminator(ml.BaseModel):
#     def __init__(
#         self,
#         rates: list = [],
#         periods: list = [2, 3, 5, 7, 11],
#         fft_sizes: list = [2048, 1024, 512],
#         sample_rate: int = 44100,
#         bands: list = BANDS,
#     ):
#         """Discriminator that combines multiple discriminators.

#         Parameters
#         ----------
#         rates : list, optional
#             sampling rates (in Hz) to run MSD at, by default []
#             If empty, MSD is not used.
#         periods : list, optional
#             periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
#         fft_sizes : list, optional
#             Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
#         sample_rate : int, optional
#             Sampling rate of audio in Hz, by default 44100
#         bands : list, optional
#             Bands to run MRD at, by default `BANDS`
#         """
#         super().__init__()
#         discs = []
#         discs += [MPD(p) for p in periods]
#         discs += [MSD(r, sample_rate=sample_rate) for r in rates]
#         discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
#         self.discriminators = nn.ModuleList(discs)

#     def preprocess(self, y):
#         # Remove DC offset
#         y = y - y.mean(dim=-1, keepdims=True)
#         # Peak normalize the volume of input audio
#         y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
#         return y

#     def forward(self, x):
#         x = self.preprocess(x)
#         fmaps = [d(x) for d in self.discriminators]
#         return fmaps


# if __name__ == "__main__":
#     disc = Discriminator()
#     x = torch.zeros(1, 1, 44100)
#     results = disc(x)
#     for i, result in enumerate(results):
#         print(f"disc{i}")
#         for i, r in enumerate(result):
#             print(r.shape, r.mean(), r.min(), r.max())
#         print()
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal, STFTParams
from audiotools import ml
from einops import rearrange
from torch.nn.utils import weight_norm
import torchaudio
import nnAudio.features as features
from munch import Munch


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class NormConv2d(nn.Module):
    """Conv2d with normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, norm="weight_norm"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation, groups, bias)
        if norm == "weight_norm":
            self.conv = weight_norm(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class MPD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = WNConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []
        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList([
            WNConv1d(1, 16, 15, 1, padding=7),
            WNConv1d(16, 64, 41, 4, groups=4, padding=20),
            WNConv1d(64, 256, 41, 4, groups=16, padding=20),
            WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
            WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
            WNConv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []
        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
        super().__init__()
        self.cfg = cfg
        self.filters = cfg.filters
        self.max_filters = cfg.max_filters
        self.filters_scale = cfg.filters_scale
        self.kernel_size = (3, 9)
        self.dilations = cfg.dilations
        self.stride = (1, 2)
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.fs = cfg.sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                NormConv2d(
                    self.in_channels * 2,  # Real + Imaginary
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                    norm="weight_norm",
                )
            )

        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min((self.filters_scale ** (i + 1)) * self.filters, self.max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    norm="weight_norm",
                )
            )
            in_chs = out_chs
            
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                norm="weight_norm",
            )
        )

        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            norm="weight_norm",
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = torchaudio.transforms.Resample(
            orig_freq=self.fs, new_freq=self.fs * 2
        )

    def forward(self, x):
        fmap = []
        x = self.resample(x)
        z = self.cqt_transform(x)
        

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = rearrange(z, "b c w t -> b c t w")

        latent_z = []
        for i in range(self.n_octaves):
            octave_band = z[:, :, :, i * self.bins_per_octave : (i + 1) * self.bins_per_octave]
            processed_band = self.conv_pres[i](octave_band)
            latent_z.append(processed_band)
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)
        fmap.append(latent_z)
        
        return fmap


class MultiScaleSubbandCQT(nn.Module):
    """CQT discriminator at multiple scales"""
    def __init__(self, sample_rate=44100):
        super().__init__()
        cfg = Munch({
            "hop_lengths": [1024, 512, 512],
            "sampling_rate": sample_rate,
            "filters": 32,
            "max_filters": 1024,
            "filters_scale": 1,
            "dilations": [1, 2, 4],
            "in_channels": 1,
            "out_channels": 1,
            "n_octaves": [10, 10, 10],
            "bins_per_octaves": [24, 36, 48],
        })
        self.cfg = cfg
        self.discriminators = nn.ModuleList([
            DiscriminatorCQT(
                cfg,
                hop_length=cfg.hop_lengths[i],
                n_octaves=cfg.n_octaves[i],
                bins_per_octave=cfg.bins_per_octaves[i],
            )
            for i in range(len(cfg.hop_lengths))
        ])

    def forward(self, x):
        fmap = []
        for disc in self.discriminators:
            fmap.extend(disc(x))
        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

class MRD(nn.Module):
    def __init__(self, window_length: int, hop_factor: float = 0.25, 
                 sample_rate: int = 44100, bands: list = BANDS):
        """Multi-resolution spectrogram discriminator."""
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList([
            WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
            WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        ])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class Discriminator(ml.BaseModel):
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
    ):
        """Discriminator combining MPD, MSD, MRD and CQT.
        
        Parameters
        ----------
        rates : list, optional
            Sampling rates for MSD, by default []
        periods : list, optional
            Periods for MPD, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            FFT sizes for MRD, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        """
        super().__init__()
        discs = []
        # Time-domain discriminators
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        
        # Frequency-domain discriminators (both STFT and CQT)
        discs += [MRD(f, sample_rate=sample_rate) for f in fft_sizes]
        discs += [MultiScaleSubbandCQT(sample_rate=sample_rate)]
            
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps