import numpy as np
from scipy.constants import c, pi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from Waveform_Gen import prop_delay, LFM, matched_filter, zero_pad

# before padding
F_0 = 1e3
F_s = 10e3
T_s = 1/F_s
nSample = int(20/F_0/T_s)
t = np.arange(0, nSample)*T_s
x_t = np.cos(2*pi*F_0*t)
x_f = np.fft.fft(x_t)
x_f = np.fft.fftshift(x_f)
freq = np.fft.fftfreq(t.size, T_s)
freq = np.fft.fftshift(freq)


fig, ax = plt.subplots(2, 1)
ax[0].plot(t, x_t)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

ax[1].plot(freq, np.abs(x_f))
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Amp")
ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

# after padding
x_t_p, t_p, pad_s = zero_pad(x_t, t, T_s)
x_f_p = np.fft.fft(x_t_p)
x_f_p = np.fft.fftshift(x_f_p)
freq_p = np.fft.fftfreq(t_p.size, T_s)
freq_p = np.fft.fftshift(freq_p)


fig, ax = plt.subplots(2, 1)
ax[0].plot(t_p, x_t_p)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

ax[1].plot(freq_p, np.abs(x_f_p))
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Amp")
ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

plt.show()
# after padding

