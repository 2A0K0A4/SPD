"""import numpy as np
from scipy.io.wavfile import write

sample_rate = 44100
duration = 5  # seconds

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

write("test.wav", sample_rate, audio.astype(np.float32))

print("WAV file created: test.wav") """

