import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Function to create and apply a Butterworth filter


def butter_filter(data, cutoff, fs, btype, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y


# Load the audio files using librosa
file_good = '/Users/Anira/Downloads/good belt.mp3'
file_faulty = '/Users/Anira/Downloads/faulty.mp3'

# Load audio and extract sample rate
samples_good, sample_rate_good = librosa.load(file_good, sr=None)
samples_faulty, sample_rate_faulty = librosa.load(file_faulty, sr=None)

# Perform FFT on the audio samples


def perform_fft(samples, sample_rate):
    N = len(samples)
    yf = np.abs(np.fft.fft(samples))[:N//2]
    xf = np.fft.fftfreq(N, 1/sample_rate)[:N//2]
    print(N)
    return xf, yf


# Apply high-pass filter (remove frequencies below 1000 Hz)
cutoff = 1000  # Set cutoff frequency for high-pass filter
filtered_good = butter_filter(
    samples_good, cutoff, sample_rate_good, btype='high')
filtered_faulty = butter_filter(
    samples_faulty, cutoff, sample_rate_faulty, btype='high')

# Perform FFT on filtered data
xf_good_filtered, yf_good_filtered = perform_fft(
    filtered_good, sample_rate_good)
xf_faulty_filtered, yf_faulty_filtered = perform_fft(
    filtered_faulty, sample_rate_faulty)

# Plot the FFT results for both sounds on the same graph for comparison
plt.figure(figsize=(14, 7))
plt.plot(xf_faulty_filtered, yf_faulty_filtered,color='red', label='Faulty Belt')
plt.plot(xf_good_filtered, yf_good_filtered,color='green', label='Good Belt')



plt.title('Filtered Frequency Domain (High-Pass) - Good vs Faulty Belt')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(True)

# Zoom in on frequencies up to 5000 Hz to highlight differences
plt.xlim(0, 12000)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(xf_faulty_filtered, yf_faulty_filtered,color='red', label='Faulty Belt')
plt.plot(xf_good_filtered, yf_good_filtered,color='green', label='Good Belt')



plt.title('Filtered Frequency Domain (High-Pass) - Good vs Faulty Belt')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(True)

# Zoom in on frequencies up to 5000 Hz to highlight differences
plt.xlim(5000, 10000)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))

plt.plot(xf_good_filtered, yf_good_filtered,color='green', label='Good Belt')

plt.title('Filtered Frequency Domain (High-Pass) - Good Belt')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(True)

# Zoom in on frequencies up to 5000 Hz to highlight differences
plt.xlim(5000, 10000)

plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 7))

plt.plot(xf_faulty_filtered, yf_faulty_filtered,
         color='red', label='Faulty Belt')

plt.title('Filtered Frequency Domain (High-Pass) - Faulty Belt')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.grid(True)

# Zoom in on frequencies up to 5000 Hz to highlight differences
plt.xlim(5000, 10000)

plt.tight_layout()
plt.show()



