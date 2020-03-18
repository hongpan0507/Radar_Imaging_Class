import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sinc
from Sinc_Interp import *
import timer

f = 1e3     # signal freq
T = 1/f     # signal period

fs = 10.5*f   # sampling freq
Ts = 1/fs   # sampling period

n = int(4*T/Ts) + 1   # number of samples; four period worth of samples
t = np.arange(0, n, 1)*Ts   # sampled time axis
y = np.cos(2*pi*f*t) + np.cos(2*pi*3*f*t) + np.cos(2*pi*5*f*t)       # signal to be up-sampled
# y = np.cos(2*pi*f*t)

up_sampling_factor = 10
Ts_up = Ts / up_sampling_factor
n_up = up_sampling_factor * y.size
t_up_new = np.arange(0, n_up, 1) * Ts_up
y_up_new = np.cos(2*pi*f*t_up_new) + np.cos(2*pi*3*f*t_up_new) + np.cos(2*pi*5*f*t_up_new)

time = timer.Timer()
time.start()

y_up1, t_up1 = sinc_interp(y, Ts)
time.count()

y_up2, t_up2 = sinc_interp_vector1(y, Ts)
time.count()

y_up3, t_up3 = sinc_interp_vector2(y, Ts)
time.count()

y_up4, t_up4 = sinc_interp_convolve(y, Ts)
time.count()

y_up5, t_up5 = sinc_interp_vector11(y, t, Ts)
time.count()

FFT_y = np.fft.fftshift(np.fft.fft(y))
Freq_y = np.fft.fftshift(np.fft.fftfreq(y.size, Ts))

fig, ax = plt.subplots(2, 1)
ax[0].plot(Freq_y, abs(FFT_y), marker='.', color='k')
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
# ax[0].set_ylim(-1, 1)


fig, ax = plt.subplots(3, 1)

ax[0].plot(t, y, marker='.', color='k')
ax[0].plot(t_up5, y_up5, marker='.', color='r')
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
# ax[0].set_ylim(-1, 1)

ax[1].plot(t_up_new, y_up_new, marker='.', color='black')
ax[1].plot(t_up1, y_up1, marker='.', color='r')
# ax[1].plot(t_up2, y_up2, marker='.', color='b')
# ax[1].plot(t_up3, y_up3, marker='.', color='g')
# ax[1].plot(t_up4, y_up4, marker='.', color='y')
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amp")
# ax[1].set_ylim(-1, 1)

ax[2].plot(t_up_new, y_up_new, marker='.', color='black')
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Amp")
# ax[2].set_ylim(-1, 1)

plt.show()
