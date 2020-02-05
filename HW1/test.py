import numpy as np
from scipy.constants import c, pi
from scipy import signal as sig
import matplotlib.pyplot as plt
from Waveform_Gen import prop_delay


T_p = 1  # pulse width
BW = 5e2    # bandwidth
F_s = 200*BW   # ADC sampling frequency
T_s = 1/F_s     # ADC sampling period
nSample = int(T_p*F_s)   # number of samples from ADC
chp_rt = BW/T_p   # chirp rate
t = np.arange(0, nSample, 1)*T_s      # (start, stop, step)
phase = -2*pi*BW/2*t+pi*chp_rt*np.sqrt(t)
freq = -BW/2+chp_rt*t
x_t = np.cos(phase)    # un-windowed signal

# win_len = int(t.size*0.2)
# window = sig.hamming(win_len)
# pad_w = int((t.size - win_len)/2)
# window = np.pad(window, (pad_w, pad_w), 'constant', constant_values=0)      # pad the size of the window to be the same as t
#
# x_t = x_t*window    # windowed signal

fig = plt.figure()
plt.suptitle("LFM" + ", Chirp Rate = " + str(chp_rt) + "Hz/s, BW = " + str(BW) + "Hz, PW = " + str(T_p) + "s")

ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, x_t)
ax1.set_xlabel("Time (s)")
plt.title("Waveform")

ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, phase)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (Radian)")
plt.title("Phase")

ax3 = plt.subplot(3, 1, 3)
ax3.plot(t, freq)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Freq (Hz)")
plt.title("Frequency")

plt.subplots_adjust(hspace=0.5)
plt.show()
