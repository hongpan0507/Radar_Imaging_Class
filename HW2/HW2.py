import numpy as np
from scipy.constants import c, pi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from matplotlib import cm as cm
from Waveform_Gen import prop_delay, LFM, matched_filter, zero_pad


# ADC parameters; PRI
F_D = 0     # doppler shift
T_p = 5e-6      # pulse width
BW = 50e6       # bandwidth
F_s = 2*BW      # ADC sampling frequency
T_s = 1/F_s     # ADC sampling period
T_PRI = 1e-3    # Pulse repetition interval
pulse_num = 100     # number of pulses

# Physical parameters
R_min = 0     # minimum range of interest
R_max = 10e3    # maximum range of interest
t_i = 2*R_min/c     # time to reach to minimum range and back
t_f = 2*R_max/c     # time to reach to max range and back
R_t0 = 5e3       # target range; unit = m
target_v = 30            # target velocity; unit = m/s; 30m/s=108km/h=67mph
G = 100    # Antenna Gain = 20dB
F_0 = 10e9   # Carrier frequency; unit = Hz
w_len = c/F_0   # wave length; unit = m
RCS = 10        # radar cross section
pha_rand = np.random.normal(0, 1, 1)*2*pi      # random.normal(mean, std, size); constant random phase factor
RN_mean = 0
RN_std = 10e-9 * 0.1    # noise power

x_t, t = LFM(BW, F_s, T_p, plot=False)    # un-windowed TX waveform
x_t_w = x_t * sig.windows.hann(t.size)       # windowed TX waveform

t_tol = T_p + t_f + T_p         # time(pulse completely out of TX) + time travel + time(pulse completely back to RX)
nSample = int(t_tol/T_s)        # ADC samples
t_abs = np.arange(0, nSample, 1)*T_s    # absolute time scale for building RX signal
R_abs = t_abs*c/2   # range
x_t_abs = np.zeros(nSample, dtype=complex)             # building RX signal
x_t_abs[0:t.size] = x_t_abs[0:t.size] + x_t_w   # add windowed waveform to the signal to simulate TX signal

# sending pulses and receive echos
for m in range(0, pulse_num):
    R_t = R_t0 - m*target_v*T_PRI   # target location at each pulse
    tau = 2 * R_t / c  # propagation time delay; unit = s
    # print(tau)
    alpha = np.sqrt((G**2 * w_len**2 * RCS) / ((4*pi)**3 * R_t**4))     # Radar equation
    # time delayed signal or received waveform and convert into two dimension array
    x_t_rx_temp = alpha * np.exp(1j*pha_rand) * prop_delay(x_t_abs, t_abs, tau, t_i, F_s)
    RN_I = np.random.normal(RN_mean, RN_std, x_t_rx_temp.size)  # adding white noise; scaled down based on the RX power; affects I
    RN_Q = np.random.normal(RN_mean, RN_std, x_t_rx_temp.size) * 1j  # adding white noise; scaled down based on the RX power; affects I
    x_t_rx_temp = x_t_rx_temp + RN_I + RN_Q  # adding noise
    x_t_rx_temp = np.array([x_t_rx_temp])
    if m == 0:
        x_t_rx = x_t_rx_temp
    else:
        x_t_rx = np.concatenate((x_t_rx, x_t_rx_temp))



# match filter for each range bin
for i in range(0, pulse_num):
    MFO = np.array([matched_filter(x_t_w, x_t_rx[i, :])])
    if i == 0:
        RangeBin = MFO
    else:
        RangeBin = np.concatenate((RangeBin, MFO))

# Plot 2D array
# RangeBin = np.transpose(RangeBin)   # flip the x, y axis for plotting only;;;; not goooooood
fig, ax = plt.subplots(1, 1)
plt.imshow(np.abs(RangeBin), cmap=cm.jet, extent=[0, t_abs.max()*c/2, 0, RangeBin.shape[0]])     # plot 2-D array and correct range scale
# plt.imshow(np.abs(RangeBin.transpose()), extent=[0, RangeBin.shape[1], 0, t_abs.max()*c/2])     # plot 2-D array and correct range scale
ax.set_title('Pulse Doppler')
ax.set_aspect('auto')
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.colorbar(orientation='vertical')

# Plot 1 slice of the 2D array for testing
fig, ax = plt.subplots(3, 1)
ax[0].plot(t, x_t_w.real, color='r')
ax[0].plot(t, x_t_w.imag, color='b')
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amp")
ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[0].set_title("TX waveform")

ax[1].plot(R_abs, x_t_rx[0].real, color='r')
ax[1].plot(R_abs, x_t_rx[0].imag, color='b')
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amp")
ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].set_title("delayed waveform")

ax[2].plot(R_abs, np.abs(RangeBin[0]))
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
