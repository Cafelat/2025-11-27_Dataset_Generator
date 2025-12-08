from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import datetime
from enum import Enum, auto
from fractions import Fraction
from typing import Union, Type
import warnings

import cv2
import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size
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
        
        if isinstance(self.window, str):
            window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        elif isinstance(self.window, (list, tuple)):
            window = scipy.signal.get_window(self.window, self.win_length, fftbins=True)
        elif isinstance(self.window, function):
            window = self.window(self.win_length)
        elif isinstance(self.window, np.ndarray):
            window = self.window

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
    return librosa.power_to_db(power_spec_data, ref=ref**2)

def db_to_power(db_spec_data: np.ndarray, ref: float) -> np.ndarray:
    return librosa.db_to_power(db_spec_data, ref=ref**2)

def power_to_amplitude(power_spec_data: np.ndarray) -> np.ndarray:
    return np.sqrt(power_spec_data)

def amplitude_and_phase_to_complex(amplitude_spec_data: np.ndarray, phase_spec_data: np.ndarray) -> np.ndarray:
    return amplitude_spec_data * np.exp(1j * phase_spec_data)

def complex_to_phase(complex_spec_data: np.ndarray) -> np.ndarray:
    return np.angle(complex_spec_data)

def phase_to_sine(phase_spec_data: np.ndarray) -> np.ndarray:
    return np.sin(phase_spec_data)

def phase_to_cosine(phase_spec_data: np.ndarray) -> np.ndarray:
    return np.cos(phase_spec_data)

def sine_and_cosine_to_phase(sine_spec_data: np.ndarray, cosine_spec_data: np.ndarray) -> np.ndarray:
    return np.arctan2(sine_spec_data, cosine_spec_data)

def complex_to_real(complex_spec_data: np.ndarray) -> np.ndarray:
    return np.real(complex_spec_data)

def complex_to_imaginary(complex_spec_data: np.ndarray) -> np.ndarray:
    return np.imag(complex_spec_data)

def real_and_imaginary_to_complex(real_spec_data: np.ndarray, imaginary_spec_data: np.ndarray) -> np.ndarray:
    return real_spec_data + 1j * imaginary_spec_data

class SpectrogramType(Enum):
    COMPLEX = auto()
    AMPLITUDE = auto()
    POWER = auto()
    DB = auto()
    PHASE = auto()
    SINE = auto()
    COSINE = auto()
    REAL = auto()
    IMAGINARY = auto()

@dataclass(frozen=True)
class BaseSpectrogram(ABC):
    data: np.ndarray
    sr: int
    stft_params: Type[STFTParameters]

    @property
    @abstractmethod
    def type(self):
        pass

    @classmethod
    @abstractmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        pass

    # @abstractmethod
    # def plot(self, ax=None, **kwargs):
    #     pass

@dataclass(frozen=True)
class ComplexSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.COMPLEX

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
    

class AmplitudeSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.AMPLITUDE

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
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        amplitude_data = complex_to_amplitude(complex_spec.data)
        return cls(amplitude_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms) -> Type[TimeSeriesData]:

        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], PhaseSpectrogram):
            phase_spec = args_spectrograms[0]
            if self.sr != phase_spec.sr:
                raise ValueError("Sampling rates of amplitude and phase spectrograms must match.")
            if self.stft_params != phase_spec.stft_params:
                raise ValueError("STFT parameters of amplitude and phase spectrograms must match.")
            
            complex_data = amplitude_and_phase_to_complex(self.data, phase_spec.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 2 and (
            (isinstance(args_spectrograms[0], 'SineComponentSpectrogram') and isinstance(args_spectrograms[1], 'CosineComponentSpectrogram')) or \
            (isinstance(args_spectrograms[0], 'CosineComponentSpectrogram') and isinstance(args_spectrograms[1], 'SineComponentSpectrogram'))
        ):
            if isinstance(args_spectrograms[0], 'SineComponentSpectrogram'):
                sine_spec = args_spectrograms[0]
                cosine_spec = args_spectrograms[1]
            else:
                sine_spec = args_spectrograms[1]
                cosine_spec = args_spectrograms[0]
            
            if self.sr != sine_spec.sr or self.sr != cosine_spec.sr:
                raise ValueError("Sampling rates of amplitude, sine, and cosine spectrograms must match.")
            if self.stft_params != sine_spec.stft_params or self.stft_params != cosine_spec.stft_params:
                raise ValueError("STFT parameters of amplitude, sine, and cosine spectrograms must match.")
            
            phase_data = sine_and_cosine_to_phase(sine_spec.data, cosine_spec.data)
            complex_data = amplitude_and_phase_to_complex(self.data, phase_data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 0:
            warnings.warn("Reconstruction from amplitude spectrogram using Griffin-Lim algorithm may produce artifacts.", UserWarning, stacklevel=2)
            data = librosa.griffinlim(
                S=self.data,
                n_iter=32,
                hop_length=self.stft_params.hop_length,
                win_length=self.stft_params.win_length,
                window=self.stft_params.window,
                center=self.stft_params.center,
            )

        else:
            raise ValueError("Invalid arguments for time series reconstruction.")
        
        return TimeSeriesData(data, self.sr)
    
    def plot(self, ax=None, **kwargs):
        
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='magma', **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        )) # フレーム番号（インデックス）からフレームの最初の時刻に変換
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Amplitude Spectrogram")
        return ax
    
class PowerSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.POWER

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
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        amplitude_data = complex_to_amplitude(complex_spec.data)
        power_data = amplitude_to_power(amplitude_data)
        return cls(power_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms) -> Type[TimeSeriesData]:

        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], PhaseSpectrogram):
            phase_spec = args_spectrograms[0]
            if self.sr != phase_spec.sr:
                raise ValueError("Sampling rates of power and phase spectrograms must match.")
            if self.stft_params != phase_spec.stft_params:
                raise ValueError("STFT parameters of power and phase spectrograms must match.")
            
            amplitude_data = power_to_amplitude(self.data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_spec.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 2 and (
            (isinstance(args_spectrograms[0], 'SineComponentSpectrogram') and isinstance(args_spectrograms[1], 'CosineComponentSpectrogram')) or \
            (isinstance(args_spectrograms[0], 'CosineComponentSpectrogram') and isinstance(args_spectrograms[1], 'SineComponentSpectrogram'))
        ):
            if isinstance(args_spectrograms[0], 'SineComponentSpectrogram'):
                sine_spec = args_spectrograms[0]
                cosine_spec = args_spectrograms[1]
            else:
                sine_spec = args_spectrograms[1]
                cosine_spec = args_spectrograms[0]
            
            if self.sr != sine_spec.sr or self.sr != cosine_spec.sr:
                raise ValueError("Sampling rates of power, sine, and cosine spectrograms must match.")
            if self.stft_params != sine_spec.stft_params or self.stft_params != cosine_spec.stft_params:
                raise ValueError("STFT parameters of power, sine, and cosine spectrograms must match.")
            
            amplitude_data = power_to_amplitude(self.data)
            phase_data = sine_and_cosine_to_phase(sine_spec.data, cosine_spec.data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 0:
            warnings.warn("Reconstruction from power spectrogram using Griffin-Lim algorithm may produce artifacts.", UserWarning, stacklevel=2)
            amplitude_data = power_to_amplitude(self.data)
            amplitude_spec = AmplitudeSpectrogram(amplitude_data, self.sr, self.stft_params)
            return amplitude_spec.to_time_series_data()
    
        else:
            raise ValueError("Invalid arguments for time series reconstruction.")
        
    def plot(self, ax=None, **kwargs):
        
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='magma', **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        ))  # インデックス番号から時間に変換

        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Power Spectrogram')
        
        return ax
    
class DBSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.DB
    
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
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        amplitude_data = complex_to_amplitude(complex_spec.data)
        power_data = amplitude_to_power(amplitude_data)
        db_data = power_to_db(power_data, ref=stft_params.ref)
        return cls(db_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms):
        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], PhaseSpectrogram):
            phase_spec = args_spectrograms[0]
            if self.sr != phase_spec.sr:
                raise ValueError("Sampling rates of dB and phase spectrograms must match.")
            if self.stft_params != phase_spec.stft_params:
                raise ValueError("STFT parameters of dB and phase spectrograms must match.")
            
            power_data = db_to_power(self.data, ref=self.stft_params.ref)
            amplitude_data = power_to_amplitude(power_data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_spec.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 2 and (
            (isinstance(args_spectrograms[0], 'SineComponentSpectrogram') and isinstance(args_spectrograms[1], 'CosineComponentSpectrogram')) or \
            (isinstance(args_spectrograms[0], 'CosineComponentSpectrogram') and isinstance(args_spectrograms[1], 'SineComponentSpectrogram'))
        ):
            if isinstance(args_spectrograms[0], 'SineComponentSpectrogram'):
                sine_spec = args_spectrograms[0]
                cosine_spec = args_spectrograms[1]
            else:
                sine_spec = args_spectrograms[1]
                cosine_spec = args_spectrograms[0]
            
            if self.sr != sine_spec.sr or self.sr != cosine_spec.sr:
                raise ValueError("Sampling rates of dB, sine, and cosine spectrograms must match.")
            if self.stft_params != sine_spec.stft_params or self.stft_params != cosine_spec.stft_params:
                raise ValueError("STFT parameters of dB, sine, and cosine spectrograms must match.")
            
            power_data = db_to_power(self.data, ref=self.stft_params.ref)
            amplitude_data = power_to_amplitude(power_data)
            phase_data = sine_and_cosine_to_phase(sine_spec.data, cosine_spec.data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 0:
            warnings.warn("Reconstruction from dB spectrogram using Griffin-Lim algorithm may produce artifacts.", UserWarning, stacklevel=2)
            power_data = db_to_power(self.data, ref=self.stft_params.ref)
            amplitude_data = power_to_amplitude(power_data)
            amplitude_spec = AmplitudeSpectrogram(amplitude_data, self.sr, self.stft_params)
            return amplitude_spec.to_time_series_data()
        
        else:
            raise ValueError("Invalid arguments for time series reconstruction.")

    def plot(self, ax=None, **kwargs):
        
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='magma', vmin=-128, vmax=0, **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        )) # フレーム番号（インデックス）からフレームの最初の時刻に変換

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Magnitude (dB)")

        return ax


class PhaseSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.PHASE
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        phase_data = np.angle(complex_spec.data)
        return cls(phase_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_sine_and_cosine(cls, sine_spec: Type['SineComponentSpectrogram'], cosine_spec: Type['CosineComponentSpectrogram']):
        if sine_spec.sr != cosine_spec.sr:
            raise ValueError("Sampling rates of sine and cosine spectrograms must match.")
        if sine_spec.stft_params != cosine_spec.stft_params:
            raise ValueError("STFT parameters of sine and cosine spectrograms must match.")
        
        phase_data = sine_and_cosine_to_phase(sine_spec.data, cosine_spec.data)
        return cls(phase_data, sine_spec.sr, sine_spec.stft_params)
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        phase_data = np.angle(complex_spec.data)
        return cls(phase_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms) -> Type[TimeSeriesData]:
        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], AmplitudeSpectrogram):
            amplitude_spec = args_spectrograms[0]
            if self.sr != amplitude_spec.sr:
                raise ValueError("Sampling rates of phase and amplitude spectrograms must match.")
            if self.stft_params != amplitude_spec.stft_params:
                raise ValueError("STFT parameters of phase and amplitude spectrograms must match.")
            
            complex_data = amplitude_and_phase_to_complex(amplitude_spec.data, self.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], PowerSpectrogram):
            power_spec = args_spectrograms[0]
            if self.sr != power_spec.sr:
                raise ValueError("Sampling rates of phase and power spectrograms must match.")
            if self.stft_params != power_spec.stft_params:
                raise ValueError("STFT parameters of phase and power spectrograms must match.")
            
            amplitude_data = power_to_amplitude(power_spec.data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, self.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        elif len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], DBSpectrogram):
            db_spec = args_spectrograms[0]
            if self.sr != db_spec.sr:
                raise ValueError("Sampling rates of phase and dB spectrograms must match.")
            if self.stft_params != db_spec.stft_params:
                raise ValueError("STFT parameters of phase and dB spectrograms must match.")
            
            power_data = db_to_power(db_spec.data, ref=db_spec.stft_params.ref)
            amplitude_data = power_to_amplitude(power_data)
            complex_data = amplitude_and_phase_to_complex(amplitude_data, self.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        else:
            raise ValueError("Invalid arguments for time series reconstruction.")
    
    def plot(self, ax=None, **kwargs):
        
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi, **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.formatter = FuncFormatter(radian_formatter)
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        )) # フレーム番号（インデックス）からフレームの最初の時刻に変換

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Phase (radians)")

        return ax
    
class SineComponentSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.SINE_COMPONENT
    
    @classmethod
    def from_phase_spectrogram(cls, phase_spec: Type[PhaseSpectrogram]):
        sine_data = phase_to_sine(phase_spec.data)
        return cls(sine_data, phase_spec.sr, phase_spec.stft_params)
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        phase_data = complex_to_phase(complex_spec.data)
        sine_data = phase_to_sine(phase_data)
        return cls(sine_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        phase_data = complex_to_phase(complex_spec.data)
        sine_data = phase_to_sine(phase_data)
        return cls(sine_data, ts_data.sr, stft_params)

    def to_time_series_data(self, *args_spectrograms):
        if len(args_spectrograms) == 2 and (
            (isinstance(args_spectrograms[0], 'CosineComponentSpectrogram') and (isinstance(args_spectrograms[1], AmplitudeSpectrogram) or isinstance(args_spectrograms[1], PowerSpectrogram) or isinstance(args_spectrograms, DBSpectrogram))) or \
            (isinstance(args_spectrograms[1], 'CosineComponentSpectrogram') and (isinstance(args_spectrograms[0], AmplitudeSpectrogram) or isinstance(args_spectrograms[0], PowerSpectrogram) or isinstance(args_spectrograms, DBSpectrogram)))
        ):
            if isinstance(args_spectrograms[0], 'CosineComponentSpectrogram'):
                cosine_spec = args_spectrograms[0]
                other_spec = args_spectrograms[1]
            else:
                cosine_spec = args_spectrograms[1]
                other_spec = args_spectrograms[0]
            
            if self.sr != cosine_spec.sr:
                raise ValueError("Sampling rates of sine and cosine spectrograms must match.")
            if self.stft_params != cosine_spec.stft_params:
                raise ValueError("STFT parameters of sine and cosine spectrograms must match.")
            
            phase_data = sine_and_cosine_to_phase(self.data, cosine_spec.data)
            
            if isinstance(other_spec, AmplitudeSpectrogram):
                amplitude_data = other_spec.data
            elif isinstance(other_spec, PowerSpectrogram):
                amplitude_data = power_to_amplitude(other_spec.data)
            elif isinstance(other_spec, DBSpectrogram):
                power_data = db_to_power(other_spec.data, ref=other_spec.stft_params.ref)
                amplitude_data = power_to_amplitude(power_data)
            else:
                raise ValueError("Invalid spectrogram type for reconstruction.")
            
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()

        else:
            raise ValueError("Invalid arguments for time series reconstruction.")

class CosineComponentSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.COSINE_COMPONENT
    
    @classmethod
    def from_phase_spectrogram(cls, phase_spec: Type[PhaseSpectrogram]):
        cosine_data = phase_to_cosine(phase_spec.data)
        return cls(cosine_data, phase_spec.sr, phase_spec.stft_params)
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        phase_data = complex_to_phase(complex_spec.data)
        cosine_data = phase_to_cosine(phase_data)
        return cls(cosine_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        phase_data = complex_to_phase(complex_spec.data)
        cosine_data = phase_to_cosine(phase_data)
        return cls(cosine_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms):
        if len(args_spectrograms) == 2 and (
            (isinstance(args_spectrograms[0], 'SineComponentSpectrogram') and (isinstance(args_spectrograms[1], AmplitudeSpectrogram) or isinstance(args_spectrograms[1], PowerSpectrogram) or isinstance(args_spectrograms, DBSpectrogram))) or \
            (isinstance(args_spectrograms[1], 'SineComponentSpectrogram') and (isinstance(args_spectrograms[0], AmplitudeSpectrogram) or isinstance(args_spectrograms[0], PowerSpectrogram) or isinstance(args_spectrograms, DBSpectrogram)))
        ):
            if isinstance(args_spectrograms[0], 'SineComponentSpectrogram'):
                sine_spec = args_spectrograms[0]
                other_spec = args_spectrograms[1]
            else:
                sine_spec = args_spectrograms[1]
                other_spec = args_spectrograms[0]
            
            if self.sr != sine_spec.sr:
                raise ValueError("Sampling rates of sine and cosine spectrograms must match.")
            if self.stft_params != sine_spec.stft_params:
                raise ValueError("STFT parameters of sine and cosine spectrograms must match.")
            
            phase_data = sine_and_cosine_to_phase(sine_spec.data, self.data)
            
            if isinstance(other_spec, AmplitudeSpectrogram):
                amplitude_data = other_spec.data
            elif isinstance(other_spec, PowerSpectrogram):
                amplitude_data = power_to_amplitude(other_spec.data)
            elif isinstance(other_spec, DBSpectrogram):
                power_data = db_to_power(other_spec.data, ref=other_spec.stft_params.ref)
                amplitude_data = power_to_amplitude(power_data)
            else:
                raise ValueError("Invalid spectrogram type for reconstruction.")
            
            complex_data = amplitude_and_phase_to_complex(amplitude_data, phase_data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()

        else:
            raise ValueError("Invalid arguments for time series reconstruction.")

class RealPartSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.REAL_PART
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        real_data = complex_to_real(complex_spec.data)
        return cls(real_data, complex_spec.sr, complex_spec.stft_params)

    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        real_data = complex_to_real(complex_spec.data)
        return cls(real_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms):
        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], ImaginaryPartSpectrogram):
            imaginary_spec = args_spectrograms[0]
            if self.sr != imaginary_spec.sr:
                raise ValueError("Sampling rates of real and imaginary part spectrograms must match.")
            if self.stft_params != imaginary_spec.stft_params:
                raise ValueError("STFT parameters of real and imaginary part spectrograms must match.")
            
            complex_data = real_and_imaginary_to_complex(self.data, imaginary_spec.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        else:
            raise ValueError("Invalid arguments for time series reconstruction.")
        
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='magma', **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        ))  # インデックス番号から時間に変換
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Real Part Spectrogram')
        return ax

class ImaginaryPartSpectrogram(BaseSpectrogram):

    @property
    def type(self):
        return SpectrogramType.IMAGINARY_PART
    
    @classmethod
    def from_complex_spectrogram(cls, complex_spec: Type[ComplexSpectrogram]):
        imaginary_data = complex_to_imaginary(complex_spec.data)
        return cls(imaginary_data, complex_spec.sr, complex_spec.stft_params)
    
    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters]):
        complex_spec = ComplexSpectrogram.from_time_series_data(ts_data, stft_params)
        imaginary_data = complex_to_imaginary(complex_spec.data)
        return cls(imaginary_data, ts_data.sr, stft_params)
    
    def to_time_series_data(self, *args_spectrograms):
        if len(args_spectrograms) == 1 and isinstance(args_spectrograms[0], RealPartSpectrogram):
            real_spec = args_spectrograms[0]
            if self.sr != real_spec.sr:
                raise ValueError("Sampling rates of imaginary and real part spectrograms must match.")
            if self.stft_params != real_spec.stft_params:
                raise ValueError("STFT parameters of imaginary and real part spectrograms must match.")
            
            complex_data = real_and_imaginary_to_complex(real_spec.data, self.data)
            complex_spec = ComplexSpectrogram(complex_data, self.sr, self.stft_params)
            return complex_spec.to_time_series_data()
        
        else:
            raise ValueError("Invalid arguments for time series reconstruction.")
    
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        img = ax.imshow(self.data, aspect='equal', origin='lower', cmap='magma', **kwargs)

        # カラーバーを設定
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        cbar = plt.colorbar(ax.images[0], cax=cax, orientation='vertical')
        cbar.update_ticks()

        # y軸の設定 
        frequency_labels = librosa.fft_frequencies(sr=self.sr, n_fft=self.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())  # 取得したい周波数目盛り
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y軸の目盛り位置を設定
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(self.data.shape[1]), sr=self.sr, hop_length=self.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=self.sr, hop_length=self.stft_params.hop_length))))}"
        ))  # インデックス番号から時間に変換

        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Imaginary Part Spectrogram')
        return ax
    
class Spectrogram:
    _registry = {
        SpectrogramType.COMPLEX: ComplexSpectrogram,
        SpectrogramType.AMPLITUDE: AmplitudeSpectrogram,
        SpectrogramType.POWER: PowerSpectrogram,
        SpectrogramType.DB: DBSpectrogram,
        SpectrogramType.PHASE: PhaseSpectrogram,
        SpectrogramType.SINE: SineComponentSpectrogram,
        SpectrogramType.COSINE: CosineComponentSpectrogram,
        SpectrogramType.REAL: RealPartSpectrogram,
        SpectrogramType.IMAGINARY: ImaginaryPartSpectrogram,
    }

    @classmethod
    def from_time_series_data(cls, ts_data: Type[TimeSeriesData], stft_params: Type[STFTParameters], spec_type: SpectrogramType):
        if spec_type not in cls._registry:
            raise ValueError(f"Unsupported spectrogram type: {spec_type}")
        spectrogram_class = cls._registry[spec_type]
        return spectrogram_class.from_time_series_data(ts_data, stft_params)


class SpectrogramVisualizer:
    @staticmethod
    def plot_DB_and_phase(db_spec: Type[DBSpectrogram], phase_spec: Type[PhaseSpectrogram], graph_ax=None, **kwargs):
        if phase_spec.sr != db_spec.sr:
            raise ValueError("Sampling rates of phase and dB spectrograms must match.")
        if phase_spec.stft_params != db_spec.stft_params:
            raise ValueError("STFT parameters of phase and dB spectrograms must match.")
        if phase_spec.data.shape != db_spec.data.shape:
            raise ValueError("Shapes of phase and dB spectrograms must match.")
        
        phase_data = phase_spec.data.copy()
        db_data = db_spec.data.copy()

        hue = (phase_data + np.pi) / (2 * np.pi) * 179  # 0-179に正規化
        saturation = (db_data -(-128)) / (0 - (-128)) * 255  # 0-255に正規化
        value = (db_data -(-128)) / (0 - (-128)) * 255  # 0-255に正規化

        hsv_image = np.stack((hue, saturation, value), axis=-1).astype(np.uint8)
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        graph_ax = graph_ax or plt.gca()
        graph_img = graph_ax.imshow(rgb_image, aspect='equal', origin='lower', **kwargs)

        # y軸の設定
        frequency_labels = librosa.fft_frequencies(sr=phase_spec.sr, n_fft=phase_spec.stft_params.n_fft)                # y軸の周波数
        frequency_ticks = ticker.AutoLocator().tick_values(frequency_labels.min(), frequency_labels.max())
        if frequency_ticks[-1] > frequency_labels.max():                                                    # 最大値を超える場合は置き換え
            frequency_ticks[-1] = frequency_labels.max()
        frequency_indices = np.searchsorted(frequency_labels, frequency_ticks)                              # 取得したい目盛りをインデックス番号に変換
        graph_ax.yaxis.set_major_locator(ticker.FixedLocator(frequency_indices))                                  # y
        graph_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(frequency_labels[int(x)])}"))      # インデックス番号から周波数に変換

        # x軸の設定
        time_labels = librosa.frames_to_time(np.arange(phase_spec.data.shape[1]), sr=phase_spec.sr, hop_length=phase_spec.stft_params.hop_length)  # x軸の時間
        time_ticks = ticker.AutoLocator().tick_values(time_labels.min(), time_labels.max())[:-1]                                 # 取得したい時間目盛り
        time_indices = np.searchsorted(time_labels, time_ticks)                                                                  # 取得したい目盛りをインデックス番号に変換
        graph_ax.xaxis.set_major_locator(ticker.FixedLocator(time_indices))                                                            # x軸の目盛り位置を設定
        graph_ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: f"{str(datetime.timedelta(seconds=round(librosa.frames_to_time(x, sr=phase_spec.sr, hop_length=phase_spec.stft_params.hop_length))))}"
        ))  # インデックス番号から時間に変換
        
        graph_ax.set_xlabel("Time (s)")
        graph_ax.set_ylabel("Frequency (Hz)")
        graph_ax.set_title("Phase and Magnitude (dB) Spectrogram")

        # カラーバーの設定
        divider = make_axes_locatable(graph_ax)
        color_ax = divider.append_axes("right", size=divider.get_vertical()[0], pad=0.3)
        color_ax.set_aspect('equal')

        height = phase_spec.data.shape[0]
        width = phase_spec.data.shape[0]

        # x軸: Hue（0〜180）
        H = np.linspace(0, 180, width)[None, :].repeat(height, axis=0)

        # y軸: Saturation / Value（0〜255）
        S = np.linspace(0, 255, height)[:, None].repeat(width, axis=1)
        V = np.linspace(0, 255, height)[:, None].repeat(width, axis=1)

        # 3チャンネルにまとめる
        hsv_colorbar = np.stack([H, S, V], axis=-1)  
        bgr_colorbar = cv2.cvtColor((hsv_colorbar).astype(np.uint8), cv2.COLOR_HSV2BGR)
        rgb_colorbar = cv2.cvtColor(bgr_colorbar, cv2.COLOR_BGR2RGB)

        color_ax.imshow(rgb_colorbar, aspect='auto', origin='lower')
        color_ax.yaxis.tick_right()                 # 目盛り（ticks）を右側へ
        color_ax.yaxis.set_label_position("right")  # ラベルも右へ

        # y軸の設定: dB値
        db_labels = np.linspace(-128, 0, height)
        db_ticks = ticker.AutoLocator().tick_values(db_labels.min(), db_labels.max())
        if db_ticks[0] < db_labels.min():  # 最小値を下回る場合は削除
            db_ticks = db_ticks[1:]
        db_indices = np.searchsorted(db_labels, db_ticks)  # 取得したい目盛りをインデックス番号に変換
        color_ax.yaxis.set_major_locator(ticker.FixedLocator(db_indices))
        color_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(db_labels[int(x)])}"))  # インデックス番号からdBに変換

        phase_labels = np.linspace(-np.pi, np.pi, width)
        phase_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        phase_indices = np.searchsorted(phase_labels, phase_ticks)  # 取得したい目盛りをインデックス番号に変換
        color_ax.xaxis.set_major_locator(ticker.FixedLocator(phase_indices))
        color_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: radian_formatter(phase_labels[int(x)], pos)))  # インデックス番号からラジアンに変換

        color_ax.set_xlabel("Phase (radians)")
        color_ax.set_ylabel("Magnitude (dB)")

        return graph_ax, color_ax


@FuncFormatter
def radian_formatter(x, pos):
    frac = Fraction(x / np.pi).limit_denominator(8)
    n, d = frac.numerator, frac.denominator

    if x == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)

    if d == 1:
        if n == 1:
            return rf"${sign}\pi$"
        else:
            return rf"${sign}{n}\pi$"
    else:
        if n == 1:
            return rf"${sign}\frac{{\pi}}{{{d}}}$"
        else:
            return rf"${sign}\frac{{{n}\pi}}{{{d}}}$"


class FrequencyBandType(Enum):
    LOW = auto()
    HIGH = auto()

@dataclass(frozen=True)
class PatchSetParameters:
    spectrogram_types: list[SpectrogramType]
    size: tuple[int, int] = (256, 256)
    overlap_rate: float = 0.5
    frequency_band: FrequencyBandType = FrequencyBandType.LOW
    padding: bool = False

    def __post_init__(self):
        if len(self.spectrogram_types) == 0:
            raise ValueError("At least one spectrogram type must be specified.")
        elif len(set(self.spectrogram_types) & {SpectrogramType.COMPLEX}) == 1:
            raise ValueError("Complex spectrogram cannot be used in patch set.")
        elif len(self.spectrogram_types) == 1 and (not set(self.spectrogram_types).isdisjoint({
            SpectrogramType.PHASE, SpectrogramType.REAL, SpectrogramType.IMAGINARY})):
            raise ValueError("At least two spectrogram types must be specified when using phase, real, or imaginary spectrograms.")
        elif len(self.spectrogram_types) == 1 and (self.spectrogram_types[0] == SpectrogramType.SINE or self.spectrogram_types[0] == SpectrogramType.COSINE):
            raise ValueError("At least three spectrogram types must be specified when using sine and cosine spectrograms.")
        elif len(self.spectrogram_types) == 2 and (not set(self.spectrogram_types).isdisjoint({SpectrogramType.SINE, SpectrogramType.COSINE})):
            raise ValueError("At least three spectrogram types must be specified when using sine and cosine spectrograms.")
        elif len(self.spectrogram_types) == 2 and (len(set(self.spectrogram_types) & {SpectrogramType.REAL, SpectrogramType.IMAGINARY}) == 1):
            raise ValueError("Both real and imaginary spectrograms must be specified when using either one.")
        elif len(self.spectrogram_types) == 2 and (len(set(self.spectrogram_types) & {SpectrogramType.PHASE}) == 1) and (len(set(self.spectrogram_types) & {SpectrogramType.AMPLITUDE, SpectrogramType.POWER, SpectrogramType.DB}) == 0):
            raise ValueError("Phase spectrogram must be used with amplitude, power, or dB spectrograms.")
        elif len(self.spectrogram_types) == 2 and (len(set(self.spectrogram_types) & {SpectrogramType.DB, SpectrogramType.POWER, SpectrogramType.AMPLITUDE}) == 2):
            raise ValueError("Only one of amplitude, power, or dB spectrograms can be specified at a time.")
        elif len(self.spectrogram_types) == 3 and (not set(self.spectrogram_types).isdisjoint({SpectrogramType.SINE, SpectrogramType.COSINE})) and (len(set(self.spectrogram_types) & {SpectrogramType.AMPLITUDE, SpectrogramType.POWER, SpectrogramType.DB}) == 0):
            raise ValueError("Sine and cosine spectrograms must be used with amplitude, power, or dB spectrograms.")
        elif len(self.spectrogram_types) >= 4:
            raise ValueError("A maximum of three spectrogram types can be specified.")
        elif len(self.spectrogram_types) != len(set(self.spectrogram_types)):
            raise ValueError("Duplicate spectrogram types are not allowed.") 
            

class SpectrogramPatchSet:
    @classmethod
    def from_spectrograms(cls, spectrograms: list[BaseSpectrogram], params: Type[PatchSetParameters]):
        for spec, type in zip(spectrograms, params.spectrogram_types):
            if spec.type != type:
                raise ValueError(f"Spectrogram type {spec} is not specified in PatchSetParameters.")
            
        