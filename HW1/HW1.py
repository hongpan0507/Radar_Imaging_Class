import numpy as np
from scipy.constants import c, pi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from Waveform_Gen import prop_delay, LFM, matched_filter, zero_pad

# R_min = 10e3     # minimum range of interest
R_max = 10e3    # maximum range of interest
# t_i = 2*R_min/c     # time to reach to minimum range and back
t_f = 2*R_max/c     # time to reach to max range and back
t_i = 0e-6     # initial sample occurring time

att = 0.8       # attenuation
F_D = 10e3     # doppler shift
T_p = 5e-6      # pulse width
BW = 50e6       # bandwidth
F_s = 2*BW      # ADC sampling frequency
T_s = 1/F_s     # ADC sampling period

G = 100    # Antenna Gain = 30dB
F_0 = 10e9   # 9GHz carrier frequency
w_len = c/F_0   # wave length
RCS = 10        # radar cross section
R_t = 5e3       # target range
alpha = np.sqrt((G**2 * w_len**2 * RCS) / ((4*pi)**3 * R_t**4))     # Radar equation
pha_rand = np.random.normal(0, 1, 1)*2*pi      # random.normal(mean, std, size); constant random phase factor
# RN_PWR = 8.4e-11    # noise power
RN_mean = 0
RN_std = 15e-9    # noise power
tau = 2*R_t/c      # propagation time delay
print(tau)

x_t, t = LFM(BW, F_s, T_p, plot=False)    # un-windowed TX waveform
x_t_w = x_t * sig.windows.hann(t.size)       # windowed TX waveform

t_tol = T_p + t_f + T_p         # time(pulse completely out of TX) + time travel + time(pulse completely back to RX)
nSample = int(t_tol/T_s)        # ADC samples
t_abs = np.arange(0, nSample, 1)*T_s    # absolute time scale for RX signal
R_abs = t_abs*c/2   # range
x_t_abs = np.zeros(nSample, dtype=complex)             # building RX signal
x_t_abs[0:t.size] = x_t_abs[0:t.size] + x_t_w   # add windowed waveform to the signal to simulate TX signal
x_t_rx = alpha * np.exp(1j*pha_rand) * prop_delay(x_t_abs, t_abs, tau, t_i, F_s)   # time delayed signal or received waveform

RN_I = np.random.normal(RN_mean, RN_std, x_t_rx.size)   # adding white noise; scaled down based on the RX power; affects I
RN_Q = np.random.normal(RN_mean, RN_std, x_t_rx.size) * 1j  # adding white noise; scaled down based on the RX power; affects I
x_t_rx = x_t_rx + RN_I + RN_Q   # adding noise

MFO = matched_filter(x_t_w, x_t_rx)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, x_t_w.real, color='r')
ax[0].plot(t, x_t_w.imag, color='b')
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[0].set_title("TX waveform")

ax[1].plot(R_abs, x_t_rx.real, color='r')
ax[1].plot(R_abs, x_t_rx.imag, color='b')
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amp")
ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].set_title("delayed waveform")

ax[2].plot(R_abs, np.abs(MFO))
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Amp")
ax[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[2].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.subplots_adjust(hspace=0.5)

# ax[3].plot(t_abs, np.gradient(np.angle(MFO), t_abs))
# ax[3].set_xlabel("Time (s)")
# ax[3].set_ylabel("Amp")
# ax[3].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
# ax[3].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
# plt.subplots_adjust(hspace=0.5)

plt.show()
