import copy
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesData:
    def __init__(self, data, sr):
        self._data = data
        self._sr = sr

    @property
    def data(self):
        return self._data
    
    @property
    def sr(self):
        return self._sr
    
    @classmethod
    def load_audio_file(cls, path, sr=None):
        data, sr = librosa.load(path, sr=sr, res_type='soxr_vhq')
        return cls(data, sr)
    
    def resample(self, target_sr):
        if self.sr == target_sr:
            return copy.deepcopy(self)
        
        resampled_data = librosa.resample(self.data, orig_sr=self.sr, target_sr=target_sr, res_type='soxr_vhq')
        return __class__(resampled_data, target_sr)
    
    @classmethod
    def superimpose(cls, signal, noise, snr_db):
        
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

    def plot(self, ax=None, duration=None, **kwargs):
        
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
    




    
    


    
    