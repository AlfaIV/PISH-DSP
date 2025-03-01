import scipy as sp, numpy as np, matplotlib.pyplot as plt
from scipy.signal import max_len_seq, hilbert, decimate, medfilt, firwin, lfilter, spline_filter, gauss_spline
import random
from math import sin, pi, sqrt, tan, cos
from scipy.interpolate import interp1d
import zipfile
import os

# with zipfile.ZipFile('datasets\gpsMax.zip', 'r') as zip_ref:
#     zip_ref.extractall('datasets')

cores = os.cpu_count()
file_path = 'datasets\gpsMax.bin\gpsMax.bin'

data = np.fromfile(file_path, dtype=np.int16)
I = data[0::2]
Q = data[1::2]  

# print("I компоненты:", I)
# print("Q компоненты:", Q)
# print("размер выборок:", len(Q))
# plt.figure(figsize=(12, 6))
# plt.scatter(I, Q, marker='.', color='b')
# plt.xlabel('I (Real)')
# plt.ylabel('Q (Imaginary)')
# plt.title('Constellation Diagram (I/Q)')
# plt.grid()
# # plt.xlim(0, 0.5) 
# plt.show()

signal = np.empty(len(I), np.complex64)
signal = [complex(I[i], Q[i]) for i in range(len(signal))]
print('Размер сигнала: ', len(signal), 'Четность: ', len(signal) % 2 == 0)

fft_result = np.fft.fft(signal, workers=cores/2)
frequencies = np.fft.fftfreq(len(signal))
magnitude = np.abs(fft_result)

# plt.figure(figsize=(12, 6))
# plt.plot(frequencies, magnitude)
# plt.title('Преобразование Фурье I/Q сигналов')
# plt.xlabel('Частота')
# plt.ylabel('Амплитуда')
# plt.grid()
# # plt.xlim(0, 0.5) 
# plt.show()