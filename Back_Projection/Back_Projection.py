import numpy as np
from numpy.fft import fft, fftshift, fftfreq, fftn, ifftn
from scipy.constants import c, pi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from matplotlib import cm as cm
from Waveform_Gen import prop_delay, LFM, matched_filter, zero_pad, propagation
import timer

# ------------------ Processor parameters ------------------------------------------------------------------------------
T_p = 10e-6          # pulse width
BW = 50e6           # LFM bandwidth
F_s = 2*BW          # ADC sampling frequency
T_s = 1/F_s         # ADC sampling period
T_PRI = 100e-6      # Pulse repetition interval; max unambigous velocity = wave_len/4/T_PRI = 75m/s
pulse_num = 4096     # number of pulses
# pulse_num = 256     # number of pulses

# -------------------- Physical parameters -----------------------------------------------------------------------------
# Range and velocity
R_min = 0           # minimum range of interest
R_max = 0.5e3        # maximum range of interest; max unambigous range = speed of light * T_PRI/2 = 15km
t_i = 2*R_min/c     # time to reach to minimum range and back
t_f = 2*R_max/c     # time to reach to max range and back
R_t0 = 0.45e3          # target range; unit = m
target_v = 60       # target velocity; unit = m/s; 30m/s=108km/h=67mph

# TX power
TX_amp = np.sqrt(1000/2)        # 1000W transmitter
G = 1000                        # Antenna Gain = 30dB
F_0 = 10e9                      # Carrier frequency; unit = Hz
wave_len = c/F_0                # wave length; unit = m
RCS = 10                        # radar cross section
pha_rand = np.random.normal(0, 1, 1)*2*pi      # random.normal(mean, std, size); constant random phase factor

# RX Noise
k_B = 1.38e-23              # Boltzmann constant; J/K
Temp_room = 290             # room temperature in K
NF = 4                      # 6dB receive noise figure
NP = k_B*Temp_room*F_s*NF   # noise power
RN_mean = 0                 # white noise mean
RN_std = np.sqrt(NP/2)      # noise power

time = timer.Timer()
time.start()
print("Create TX waveform")

# construct TX Waveform and mapping ADC sampling time to physical distance
x_t, t = LFM(BW, F_s, T_p, plot=False)      # un-windowed TX waveform
x_t_w = x_t * sig.windows.hann(t.size)       # windowed TX waveform

t_tol = T_p + t_f + T_p         # time(pulse completely out of TX) + time travel + time(pulse completely back to RX)
nSample = int(t_tol/T_s)        # ADC samples
t_abs = np.arange(0, nSample, 1)*T_s    # absolute time scale for building RX signal
R_abs = t_abs*c/2   # range
x_t_abs = np.zeros(nSample, dtype=complex)             # building RX signal
x_t_abs[0:t.size] = x_t_abs[0:t.size] + x_t_w   # add windowed waveform to the signal to simulate TX signal

time.count()
print("Simulate RX signal")

# sending pulses and receive echos
x_t_rx = np.zeros((pulse_num, nSample), dtype=complex)      # pre-allocate memory to speed up calculation
for m in range(0, pulse_num):
    R_t = R_t0 - m*target_v*T_PRI   # target location at each pulse
    tau = 2 * R_t / c  # propagation time delay; unit = s
    F_D = 2*target_v/wave_len   # doppler shift
    alpha = np.sqrt((G**2 * wave_len**2 * RCS) / ((4*pi)**3 * R_t**4))     # Radar equation
    # time delayed signal or received waveform
    x_t_rx_temp = np.exp(1j*pha_rand) * propagation(x_t_abs, t_abs, tau, t_i, F_s, alpha, F_D, F_0, TX_amp)
    RN_I = np.random.normal(RN_mean, RN_std, x_t_rx_temp.size)  # adding white noise; scaled down based on the RX power; affects I
    RN_Q = np.random.normal(RN_mean, RN_std, x_t_rx_temp.size) * 1j  # adding white noise; scaled down based on the RX power; affects I
    x_t_rx_temp = x_t_rx_temp + RN_I + RN_Q     # adding noise
    x_t_rx_temp = np.array([x_t_rx_temp])       # create 2-D array
    x_t_rx[m, :] = x_t_rx_temp

time.count()
print("match filter")

# match filter to create each range bin
RangeBin = np.zeros(x_t_rx.shape, dtype=complex)    # pre-allocate memory to speed up calculation
for i in range(0, pulse_num):
    RangeBin[i, :] = matched_filter(x_t_w, x_t_rx[i, :], True)  # true = convolution; false = frequency domain

time.count()
print("Doppler Processing")

# Doppler processing to extract target speed
Doppler = fftshift(fftn(RangeBin, axes=(0,)), axes=(0,))    # n dimensional FFT, only FFT over column (over pulses)
# Doppler = Doppler.transpose()
DopplerFreq = fftshift(fftfreq(pulse_num, T_PRI))
DopplerVelocity = DopplerFreq * wave_len/2

time.count()
print("Start plotting")

# Plot 2D array
fig, ax = plt.subplots(1, 1)
plt.imshow(np.abs(RangeBin), cmap=cm.jet, extent=[0, t_abs.max()*c/2, 0, RangeBin.shape[0]])     # plot 2-D array and correct range scale
ax.set_title('Range vs Doppler')
ax.set_xlabel("Range (m)")
ax.set_ylabel("Pulse Number")
ax.set_aspect('auto')
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.colorbar(orientation='vertical')

fig, ax = plt.subplots(1, 1)
plt.imshow(np.abs(Doppler), cmap=cm.jet, extent=[0, t_abs.max()*c/2, DopplerVelocity.max(), DopplerVelocity.min()])
ax.set_title('Range vs Velocity')
ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
ax.set_aspect('auto')
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
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
ax[1].set_xlabel("Range (m)")
ax[1].set_ylabel("Amp")
ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[1].set_title("delayed waveform")

ax[2].plot(R_abs, np.abs(RangeBin[0]))
ax[2].set_xlabel("Range (m)")
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

time.count()
print("The End")
time.stop()

plt.show()
