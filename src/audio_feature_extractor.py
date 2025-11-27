

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
    
    