import copy
from dataclasses import dataclass
from typing import Union, Type
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy

class TimeSeriesData:
    def __init__(self, data: np.ndarray, sr: int):
        self._data = data
        self._sr = sr

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @classmethod
    def load_audio_file(cls, path, sr: Union[int, None] = None):
        data, sr = librosa.load(path, sr=sr, res_type='soxr_vhq')
        return cls(data, sr)
    
    def resample(self, target_sr: int):
        if self.sr == target_sr:
            return copy.deepcopy(self)
        
        resampled_data = librosa.resample(self.data, orig_sr=self.sr, target_sr=target_sr, res_type='soxr_vhq')
        return __class__(resampled_data, target_sr)
    
    @classmethod
    def superimpose(cls, signal:Type['TimeSeriesData'], noise:Type['TimeSeriesData'], snr_db: float):
        
        if signal.sr != noise.sr:
            raise ValueError("Sampling rates of signal and noise must match.")
        
        if len(signal.data) != len(noise.data):
            warnings.warn("Signal and noise lengths differ. Truncating to the shorter length.", UserWarning, stacklevel=2)

        min_length = min(len(signal.data), len(noise.data))
        signal_data = signal.data[:min_length].copy()
        noise_data  = noise.data[:min_length].copy()
        
        signal_rms  = np.sqrt(np.mean(signal_data**2))
        noise_rms   = np.sqrt(np.mean(noise_data**2))
        
        desired_noise_rms = signal_rms / (10**(snr_db / 20))
        scaling_factor = desired_noise_rms / (noise_rms + 1e-10)
        adjusted_noise_data = noise_data * scaling_factor
        adjusted_noisy_data = signal_data + adjusted_noise_data
        adjusted_signal_data = signal_data

        max_abs = np.max(np.abs(adjusted_noisy_data))
        if max_abs > 1.0:
            adjusted_noisy_data = adjusted_noisy_data / max_abs
            adjusted_signal_data = adjusted_signal_data / max_abs
            adjusted_noise_data  = adjusted_noise_data / max_abs
        
        return cls(adjusted_noisy_data, signal.sr), cls(adjusted_signal_data, signal.sr), cls(adjusted_noise_data, signal.sr)

    def plot(self, ax=None, duration: Union[None, float, tuple] = None, **kwargs):
        
        ax = ax or plt.gca()

        if isinstance(duration, tuple):
            start_time, end_time = duration
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            plot_data = self.data[start_sample:end_sample]
            time_labels = np.linspace(start_time, end_time, num=len(plot_data))
        
        elif isinstance(duration, (int, float)):
            end_sample = int(duration * self.sr)
            plot_data = self.data[:end_sample]
            time_labels = np.linspace(0, duration, num=len(plot_data))

        else:
            plot_data = self.data
            time_labels = np.linspace(0, len(self.data) / self.sr, num=len(self.data))

        ax.plot(time_labels, plot_data, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.linspace(-1, 1, 3))
        ax.set_yticks(np.linspace(-1, 1, 9), minor=True)
        ax.grid(True, which='major', axis='y')
        ax.set_title('Time-Series Waveform')

        return ax
    


@dataclass(frozen=True)
class STFTParameters:
    n_fft: int = 512
    hop_length: int | None = None
    win_length: int | None = None
    window = 'hann'
    center: bool = True
    pad_mode = 'constant'

    def __post_init__(self):
        if self.hop_length is None:
            object.__setattr__(self, 'hop_length', self.n_fft // 4)
        if self.win_length is None:
            object.__setattr__(self, 'win_length', self.n_fft)
        elif self.win_length > self.n_fft:
            raise ValueError("win_length cannot be greater than n_fft.")
        
        if isinstance(self._window, str):
            window = librosa.filters.get_window(self._window, self._win_length, fftbins=True)
        elif isinstance(self._window, (list, tuple)):
            window = scipy.signal.get_window(self._window, self._win_length, fftbins=True)
        elif isinstance(self._window, function):
            window = self._window(self._win_length)
        elif isinstance(self._window, np.ndarray):
            window = self._window

        mean_window = np.mean(window)               # 窓の平均値
        l2_norm_window = np.linalg.norm(window)     # 窓のL2ノルム(ユークリッド距離, エネルギー)の平方根
        scaling_factor = l2_norm_window / np.sqrt(len(window))  # 窓のパワーの平方根, RMS, 実効値
        reference_amplitude = self.n_fft * mean_window * scaling_factor
        object.__setattr__(self, 'ref', reference_amplitude)


def complex_to_amplitude(complex_spec_data: np.ndarray) -> np.ndarray:
    return np.abs(complex_spec_data)

def amplitude_to_power(amplitude_spec_data: np.ndarray) -> np.ndarray:
    return amplitude_spec_data ** 2

def power_to_db(power_spec_data: np.ndarray, ref: float) -> np.ndarray:
    return librosa.power_to_db(power_spec_data, ref=ref)

def db_to_power(db_spec_data: np.ndarray, ref: float) -> np.ndarray:
    return librosa.db_to_power(db_spec_data, ref=ref)

def power_to_amplitude(power_spec_data: np.ndarray) -> np.ndarray:
    return np.sqrt(power_spec_data)

def amplitude_and_phase_to_complex(amplitude_spec_data: np.ndarray, phase_spec_data: np.ndarray) -> np.ndarray:
    return amplitude_spec_data * np.exp(1j * phase_spec_data)

class ComplexSpectrogram:
    def __init__(self, data, sr, stft_params: STFTParameters):
        self._data = data
        self._sr = sr
        self._stft_params = stft_params

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @property
    def stft_params(self):
        return self._stft_params

    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        stft_matrix = librosa.stft(
            y=ts_data.data,
            n_fft=stft_params.n_fft,
            hop_length=stft_params.hop_length,
            win_length=stft_params.win_length,
            window=stft_params.window,
            center=stft_params.center,
            pad_mode=stft_params.pad_mode
        )
        return cls(stft_matrix, ts_data.sr, stft_params)

    @classmethod
    def from_amplitude_and_phase(cls, amplitude_spec: Type['AmplitudeSpectrogram'], phase_spec: Type['PhaseSpectrogram']):
        if amplitude_spec.sr != phase_spec.sr:
            raise ValueError("Sampling rates of amplitude and phase spectrograms must match.")
        if amplitude_spec.stft_params != phase_spec.stft_params:
            raise ValueError("STFT parameters of amplitude and phase spectrograms must match.")
        
        complex_data = amplitude_and_phase_to_complex(amplitude_spec.data, phase_spec.data)
        return cls(complex_data, amplitude_spec.sr, amplitude_spec.stft_params)
    
    @classmethod
    def from_power_and_phase(cls, power_spec: Type['PowerSpectrogram'], phase_spec: Type['PhaseSpectrogram']):
        if power_spec.sr != phase_spec.sr:
            raise ValueError("Sampling rates of power and phase spectrograms must match.")
        if power_spec.stft_params != phase_spec.stft_params:
            raise ValueError("STFT parameters of power and phase spectrograms must match.")
        
        amplitude_data = power_to_amplitude(power_spec.data)
        complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_spec.data)
        return cls(complex_data, power_spec.sr, power_spec.stft_params)
    
    def to_time_series_data(self) -> Type[TimeSeriesData]:
        istft_data = librosa.istft(
            self.data,
            hop_length=self.stft_params.hop_length,
            win_length=self.stft_params.win_length,
            window=self.stft_params.window,
            center=self.stft_params.center,
        )
        return TimeSeriesData(istft_data, self.sr)
    
    @classmethod
    def from_db_and_phase(cls, db_spec: Type['DBSpectrogram'], phase_spec: Type['PhaseSpectrogram']):
        if db_spec.sr != phase_spec.sr:
            raise ValueError("Sampling rates of dB and phase spectrograms must match.")
        if db_spec.stft_params != phase_spec.stft_params:
            raise ValueError("STFT parameters of dB and phase spectrograms must match.")
        
        power_data = db_to_power(db_spec.data, ref=db_spec.stft_params.ref)
        amplitude_data = power_to_amplitude(power_data)
        complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_spec.data)
        return cls(complex_data, db_spec.sr, db_spec.stft_params)

class AmplitudeSpectrogram:
    def __init__(self, data, sr, stft_params: Type[STFTParameters]):
        self._data = data
        self._sr = sr
        self._stft_params = stft_params

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @property
    def stft_params(self):
        return self._stft_params

    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        amplitude_data = complex_to_amplitude(complex_spec.data)
        return cls(amplitude_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_power_spectrogram(cls, power_spec: Type['PowerSpectrogram']):
        amplitude_data = power_to_amplitude(power_spec.data)
        return cls(amplitude_data, power_spec.sr, power_spec.stft_params)
    
    @classmethod
    def from_db_spectrogram(cls, db_spec: Type['DBSpectrogram']):
        power_data = db_to_power(db_spec.data, ref=db_spec.stft_params.ref)
        amplitude_data = power_to_amplitude(power_data)
        return cls(amplitude_data, db_spec.sr, db_spec.stft_params)
    
    def to_time_series_data(self) -> Type[TimeSeriesData]:
        warnings.warn("Reconstruction from amplitude spectrogram using Griffin-Lim algorithm may produce artifacts.", UserWarning, stacklevel=2)
        data = librosa.griffinlim(
            S=self.data,
            n_iter=32,
            hop_length=self.stft_params.hop_length,
            win_length=self.stft_params.win_length,
            window=self.stft_params.window,
            center=self.stft_params.center,
        )
        return TimeSeriesData(data, self.sr)
    
class PowerSpectrogram:
    def __init__(self, data: np.ndarray, sr: int, stft_params: Type[STFTParameters]):
        self._data = data
        self._sr = sr
        self._stft_params = stft_params

    @property
    def data(self):
        return self._data

    @property
    def sr(self):
        return self._sr

    @property
    def stft_params(self):
        return self._stft_params

    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        amplitude_data = complex_to_amplitude(complex_spec.data)
        power_data = amplitude_to_power(amplitude_data)
        return cls(power_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_amplitude_spectrogram(cls, amplitude_spec: Type[AmplitudeSpectrogram]):
        power_data = amplitude_to_power(amplitude_spec.data)
        return cls(power_data, amplitude_spec.sr, amplitude_spec.stft_params)
    
    @classmethod
    def from_db_spectrogram(cls, db_spec: Type['DBSpectrogram']):
        power_data = db_to_power(db_spec.data, ref=db_spec.stft_params.ref)
        return cls(power_data, db_spec.sr, db_spec.stft_params)
    
class DBSpectrogram:
    def __init__(self, data: np.ndarray, sr: int, stft_params: Type[STFTParameters]):
        self._data = data
        self._sr = sr
        self._stft_params = stft_params

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @property
    def stft_params(self):
        return self._stft_params
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        amplitude_data = complex_to_amplitude(complex_spec.data)
        power_data = amplitude_to_power(amplitude_data)
        db_data = power_to_db(power_data, ref=complex_spec.stft_params.ref)
        return cls(db_data, complex_spec.sr, complex_spec.stft_params)

    @classmethod
    def from_amplitude_spectrogram(cls, amplitude_spec: Type[AmplitudeSpectrogram]):
        power_data = amplitude_to_power(amplitude_spec.data)
        db_data = power_to_db(power_data, ref=amplitude_spec.stft_params.ref)
        return cls(db_data, amplitude_spec.sr, amplitude_spec.stft_params)

    @classmethod
    def from_power_spectrogram(cls, power_spec: Type[PowerSpectrogram]):
        db_data = power_to_db(power_spec.data, ref=power_spec.stft_params.ref)
        return cls(db_data, power_spec.sr, power_spec.stft_params)
    

class PhaseSpectrogram:
    def __init__(self, data: np.ndarray, sr: int, stft_params: Type[STFTParameters]):
        self._data = data
        self._sr = sr
        self._stft_params = stft_params

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @property
    def stft_params(self):
        return self._stft_params
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        phase_data = np.angle(complex_spec.data)
        return cls(phase_data, complex_spec.sr, complex_spec.stft_params)
    


class SpectrogramVisualizer:
    pass


    

    