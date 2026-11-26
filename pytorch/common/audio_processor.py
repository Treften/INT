
import numpy as np
import torch
import librosa

from librosa.filters import mel as librosa_mel_fn
from scipy.io import wavfile

from pytorch.common.audio_preprocessing import librosa_utils


class LibrosaAudioProcessor:
    def __init__(
        self,
        type,
        sampling_rate,
        max_wave_value,
        sfft=None,
        mel=None,
        augment=False,
        aug_probs=None,
        gain_db=(-5, 5),
        noise_snr_db=(20, 35),
        max_shift_sec=0.08,
        time_stretch_range=(0.95, 1.05),
        pitch_shift_range=(-1, 1),
    ):
        self.type = type
        self.sampling_rate = sampling_rate
        self.max_wave_value = max_wave_value
        self.sfft = sfft or {
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
        }

        if mel is None:
            raise ValueError("Mel configuration is required")

        mel_fmax = None if mel["mel_fmax"] == "None" else mel["mel_fmax"]

        self.mel_basis = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.sfft["filter_length"],
            n_mels=mel["n_mel_channels"],
            fmin=mel["mel_fmin"],
            fmax=mel_fmax,
        )

        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)
        self.num_channels = mel["n_mel_channels"]

        self.augment = augment

        self.aug_probs = {
            "gain": 0.5,
            "noise": 0.3,
            "shift": 0.3,
            "stretch": 0.3,
            "pitch": 0.0,
        }

        if aug_probs is not None:
            self.aug_probs.update(aug_probs)

        self.gain_db = gain_db
        self.noise_snr_db = noise_snr_db
        self.max_shift_sec = max_shift_sec
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

    def spec2wave(self, spec):
        if isinstance(spec, torch.Tensor):
            spec = spec.detach().cpu().numpy()

        return librosa_utils.inv_melspectrogram(
            spec, self.sfft, self.inv_mel_basis
        )

    def get_num_channels(self):
        return self.num_channels

    def load_wave(self, full_path):
        return librosa.load(full_path, sr=self.sampling_rate)[0]

    def load_spec(self, spec):
        return np.load(spec)

    def save_wave(self, path, wave):
        wavfile.write(path, self.sampling_rate, self.to_int(wave))

    def to_int(self, wave):
        wave = np.asarray(wave, dtype=np.float32).copy()
        peak = max(0.01, np.max(np.abs(wave)))
        wave *= self.max_wave_value / peak
        return wave.astype(np.int16)

    def _apply(self, name):
        return np.random.random() < self.aug_probs[name]

    def _fit_length(self, wave, target_length):
        """Random crop or zero-pad to preserve original duration."""

        if len(wave) > target_length:
            start = np.random.randint(0, len(wave) - target_length + 1)
            return wave[start:start + target_length]

        if len(wave) < target_length:
            diff = target_length - len(wave)
            left = np.random.randint(0, diff + 1)
            right = diff - left
            return np.pad(wave, (left, right))

        return wave

    def augment_wave(self, wave):
        """Waveform-level augmentations."""

        wave = np.asarray(wave, dtype=np.float32).copy()

        if len(wave) == 0:
            return wave

        original_length = len(wave)

        # Random Gain.
        if self._apply("gain"):
            gain = np.random.uniform(*self.gain_db)
            wave *= 10 ** (gain / 20)

        # Gaussian Noise with random SNR.
        if self._apply("noise"):
            rms = np.sqrt(np.mean(wave ** 2))

            if rms > 1e-8:
                snr = np.random.uniform(*self.noise_snr_db)
                noise_rms = rms / (10 ** (snr / 20))

                noise = np.random.normal(
                    0, noise_rms, size=wave.shape
                ).astype(np.float32)

                wave += noise

        # Random Time Shift without circular wrapping.
        if self._apply("shift"):
            max_shift = min(
                int(self.sampling_rate * self.max_shift_sec),
                original_length - 1,
            )

            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)

                shifted = np.zeros_like(wave)

                if shift > 0:
                    shifted[shift:] = wave[:-shift]
                elif shift < 0:
                    shifted[:shift] = wave[-shift:]
                else:
                    shifted = wave

                wave = shifted

        # Time Stretch.
        if self._apply("stretch"):
            rate = np.random.uniform(*self.time_stretch_range)

            wave = librosa.effects.time_stretch(
                y=wave,
                rate=rate,
            )

            wave = self._fit_length(wave, original_length)

        # Pitch Shift.
        if self._apply("pitch"):
            steps = np.random.uniform(*self.pitch_shift_range)

            wave = librosa.effects.pitch_shift(
                y=wave,
                sr=self.sampling_rate,
                n_steps=steps,
            )

            wave = self._fit_length(wave, original_length)

        return wave.astype(np.float32)

    def get_spec(self, wave):
        mel = librosa_utils.melspectrogram(
            wave, self.sfft, self.mel_basis
        )
        return torch.FloatTensor(mel)

    def process(self, wav_path, augment=None):
        wave = self.load_wave(wav_path)

        use_augment = self.augment if augment is None else augment

        if use_augment:
            wave = self.augment_wave(wave)

        return self.get_spec(wave)
