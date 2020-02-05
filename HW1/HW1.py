import numpy as np
from scipy.constants import c, pi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from Waveform_Gen import prop_delay, LFM

R_min = 1e3     # minimum range of interest
R_max = 10e3    # maximum range of interest
Tx = 0      # ???
# t_i = 2*R_min/c     # initial sample occurring time
# t_f = 2*R_max/c + Tx     # final sample occurring time
t_i = 0     # initial sample occurring time

att = 0.8       # attenuation
F_D = 0.1e3     # doppler shift
tau = 1e-6    # time delay
T_p = 5e-6  # pulse width
BW = 500e6    # bandwidth
F_s = 2*BW   # ADC sampling frequency

#x_t, t = LFM(100, 3000, 1, plot=True)    # un-windowed signal
x_t, t = LFM(BW, F_s, T_p)    # un-windowed signal

win_len = int(t.size*0.2)
window = sig.hanning(win_len)
pad_w = int((t.size - win_len)/2)
window = np.pad(window, (pad_w, pad_w), 'constant', constant_values=0)      # pad the size of the window to be the same as t

x_t = x_t*window    # windowed signal
x_t_s = prop_delay(x_t, t, tau, t_i, F_s)   # time delayed signal
h_t = np.conjugate(np.flip(x_t))    # matched filter
MFO = np.convolve(x_t_s, h_t, 'same')      # matched filter output ?????

fig = plt.figure()
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, x_t)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amp")
ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, x_t_s)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amp")

ax3 = plt.subplot(3, 1, 3)
ax3.plot(t, np.abs(MFO))
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amp")
ax3.ticklabel_format(axis='x', style='sci')
plt.subplots_adjust(hspace=0.5)

# ax2 = plt.subplot(2, 2, 1)
# ax2.plot(freq, np.abs(x_f))
# ax2.set_xlabel("Freq (Hz)")
# ax2.set_ylabel("Amp")
#
# ax3 = plt.subplot(2, 2, 2)
# ax3.plot(freq, np.angle(x_f))
# ax3.set_xlabel("Freq (Hz)")
# ax3.set_ylabel("Phase")
#
# ax4 = plt.subplot(2, 2, 3)
# ax4.plot(freq, np.abs(x_f_s))
# ax4.set_xlabel("Freq (Hz)")
# ax4.set_ylabel("Amp")
#
# ax5 = plt.subplot(2, 2, 4)
# ax5.plot(freq, np.angle(x_f_s))
# ax5.set_xlabel("Freq (Hz)")
# ax5.set_ylabel("Phase")

plt.show()
