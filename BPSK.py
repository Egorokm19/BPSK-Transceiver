import numpy as np
from math import pi
from numpy import array, arange, abs as np_abs
from numpy import ones,sin,arange,hstack,nditer,pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math
import arlpy
import arlpy.plot
from numpy.fft import rfft, rfftfreq
from math import sqrt, ceil 
from scipy.special import erfc

# number symbol
size = 10 # duration in s
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
 
# generating a random signal sequence
a = np.random.randint(0, 2, size)
m = np.zeros(len(t), dtype=np.float32)
 
for i in range(len(t)):
    m[i] = a[math.floor(t[i])]
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
 
# Plot a random signal sequence
ax1.set_title('Generate a random n-bit binary signal', fontsize = 10)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, m, 'b')

# parameter
Amp = 0.1 # sine wave amplitude
fc = 4000 # cutoff frequency
fs = 20 * fc # sampling frequency
ts = np.arange(0, (100 * size) / fs, 1 / fs)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4) # generating BPSK
BPSK = arlpy.comms.psk(2, gray=True)
arlpy.plot.iqplot(BPSK, color='red', marker='x', title = 'BPSK')

# BPSK modulated waveform
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('BPSK modulated signal', fontsize=15)
plt.axis([0,size,-1.5, 1.5])
plt.plot(t, bpsk, 'r')

# Determine additive white gauss noise
def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.power(y, 2)) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y

# adding noise AWGN
# noise_bpsk = awgn(bpsk, 5)
noise_bpsk = np.random.normal(0,1,1000)

# BPSK modulated signal with superimposed noise signal
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('Сигнал BPSK + шум', fontsize = 15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, noise_bpsk, 'r')

# Bandpass signal generation signal with bandwidth [2000,6000]
[b11,a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 80000, 6000 * 2 / 80000], btype = 'bandpass', analog = False, output = 'ba')

# The design of the low pass filter with a cut-off frequency of the bandwidth of 2000 Hz
[b12,a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 80000), btype = 'lowpass', analog = False, output = 'ba')

# Out-of-band noise filter with bandpass filter
bandpass_out = signal.filtfilt(b11, a11, noise_bpsk)

# Coherent demodulation multiplied by a coherent carrier in phase with the same frequency
coherent_demod = bandpass_out * (coherent_carrier * 2)

# Using a low pass filter
lowpass_out = signal.filtfilt(b12, a12, coherent_demod) # ФНЧ
fig2 = plt.figure()
bx1 = fig2.add_subplot(2, 1, 1)
bx1.set_title('Carrier conversion after low pass filter', fontsize=10)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, lowpass_out, 'r')

# Signal detection
detection_bpsk = np.zeros(len(t), dtype=np.float32)
flag = np.zeros(size, dtype=np.float32)

for i in range(10):
    tempF = 0
    for j in range(100):
        tempF = tempF + lowpass_out[i * 100 + j]
    if tempF > 0:
        flag[i] = 1
    else:
        flag[i] = 0
for i in range(size):
    if flag[i] == 0:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 0
    else:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 1
            
bx2 = fig2.add_subplot(2, 1, 2)
bx2.set_title('BPSK detection', fontsize=10)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, detection_bpsk, 'r')
plt.show()

# calculate the Fourier transform. The signal is valid, so rfft must be used, it is faster than fft
npts=len(t) # input time array length
spectrum = rfft(bpsk) # BPSK spectrum
spectrum1 = rfft(noise_bpsk) # BPSK signal spectrum + noise

# Spectrum BPSK
    
plt.plot(rfftfreq(npts, 1./fs), np_abs(spectrum)/npts)
# rfftfreq will do all the work of converting the numbers of the elements of the time array to hertz
# amplitude spectrum using abs from numpy (acts on arrays elementwise)
# and therefore divide by the number of elements so that the amplitudes are in millivolts and not in Fourier sums (the constant components coincide in the generated signal and in the spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal amplitude')
plt.title('Spectrum BPSK')
plt.grid(True)
#plt.savefig("SB.png", format="png", bbox_inches="tight")
plt.show()

# Spectrum Signal + Noise

plt.plot(rfftfreq(npts, 1./fs), np_abs(spectrum1)/npts)
# rfftfreq will do all the work of converting the numbers of the elements of the time array to hertz
# amplitude spectrum using abs from numpy (acts on arrays elementwise)
#  and therefore divide by the number of elements so that the amplitudes are in millivolts and not in Fourier sums (the constant components coincide in the generated signal and in the spectrum)
spectrum_bpsk = rfft(detection_bpsk) # BPSK spectrum of the detected signal
#plt.plot(rfftfreq(npts, 1./fs), np_abs(spectrum_bpsk)/npts)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal amplitude')
plt.title('Signal spectrum + noise')
plt.grid(True)
#plt.savefig("SBN.png", format="png", bbox_inches="tight")
plt.show()

# ADD DSSS

ct = arlpy.comms.random_data(1000)
#
mt = [a*b for a,b in nditer([bpsk,ct])]
bpsk_dsss = []
for i in range(0,len(mt)):
    bpsk_dsss = bpsk_dsss+[mt[i]*Cr for Cr in coherent_carrier]
#plt.plot(rfftfreq(npts, 1./fs), np_abs(spectrum_bpsk)/npts)
plt.plot(bpsk_dsss)
plt.title('DSSS BPSK signal')
plt.grid(True)
#plt.savefig("SBND.png", format="png", bbox_inches="tight")
plt.show()

# calculate the Fourier transform. The signal is valid, so rfft must be used, it is faster than fft
spectrum_dsss = rfft(bpsk_dsss) # BPSK DSSS спектр
line2, = plt.plot(rfftfreq(npts*1000, 1./fs), np_abs(spectrum_dsss)/npts, label="DSSS", linewidth=4)
plt.legend(handles=[line2], loc=1)
#plt.ylim([0.5e-4, 1])
plt.xlim([3000, 5000])
plt.xlabel('Frequency(Hz)')
plt.ylabel('Signal amplitude')
plt.title('DSSS BPSK spectrum')
plt.grid(True)
#plt.savefig("SBNDD.png", format="png", bbox_inches="tight")
plt.show()

# counting BER

SNR_MIN   = 0
SNR_MAX   = 10
Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1)
Eb_No_lin = 10**(Eb_No_dB/10.0)  # linear SNR
# Memory allocator
Pe        = empty(shape(Eb_No_lin))
BER       = empty(shape(Eb_No_lin))
# cycle start
loop = 0
for snr in Eb_No_lin:
    No = 1.0/snr
    Pe[loop] = 0.5*erfc(sqrt(snr))
    nFrames = ceil(100.0/fs/Pe[loop])
    error_sum = 0
    scale = sqrt(No/2)

    for frame in arange(nFrames):
        # error counting
        err = where (detection_bpsk != bpsk)
        error_sum += len(err[0])
        # end of cycle

BER[loop] = error_sum/(fs*nFrames)  # SNR loop level
print ("Eb_No_dB[loop] = " + str(Eb_No_dB[loop]))
print("BER[loop] = " + str(BER[loop]))
print("Pe[loop] = " + str(Pe[loop]))
loop += 1

plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()

# use matched filter

# signal + noise
data = bpsk + noise_bpsk

plt.figure()
plt.plot(t, data, '-', color="grey")
plt.plot(t, bpsk, '-', color="red", linewidth=2)
plt.xlim(0, size)
plt.xlabel('Время')
plt.ylabel('Сигнал + шум')
plt.grid(True)
#plt.savefig("gaussian_noise_w_signal.png", format="png", bbox_inches="tight")
plt.show()
# БПФ  
bpsk_fft = np.fft.fft(bpsk)
data_fft = np.fft.fft(noise_bpsk)

# Sample rates to the Nyquist limit (fs/2)  
 
sample_freq = np.fft.fftfreq(t.shape[0], 1./fs)

# FFT construction

# getting only positive spectrum: plt.xlim(0, np.max(sample_freq)) 
# big peak plan
#plt.xlim(0, np.max(sample_freq))
plt.figure()
plt.plot(sample_freq, np.abs(bpsk_fft)/np.sqrt(fs), color="red", alpha=0.5, linewidth=4)
plt.plot(sample_freq, np.abs(data_fft)/np.sqrt(fs), color="grey")
plt.xlim(0, np.max(sample_freq))
plt.xlabel('Countdown frequency')
plt.ylabel('Power spectrum')
plt.grid(True)
#plt.savefig(".png", format="png", bbox_inches="tight")
plt.show()
# Signal power spectral density
power_data, freq_psd = plt.psd(data, Fs=fs, NFFT=fs, visible=False)
power, freq = plt.psd(bpsk, Fs=fs, NFFT=fs, visible=False)
plt.figure()
# sqrt (power_data) - amplitude spectral density
plt.plot(freq_psd, np.sqrt(power_data), 'gray') # signal + noise
plt.plot(freq, np.sqrt(power), color="red", alpha=0.5, linewidth=3) # signal

# range from 0 to the Nyquist frequency
plt.xlim(0, fs/2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude spectral density (ASD)')
plt.grid(True)
#plt.savefig("ASD.png", format="png", bbox_inches="tight")
plt.show()

# Matching FFT frequency elements with frequency elements
power_vec = np.interp(sample_freq, freq_psd[2*fc:], power_data[2*fc:])

# Apply matched filter
matched_filter = 2*np.fft.ifft(data_fft * bpsk_fft.conjugate()/power_vec)
#SNR_matched = np.sqrt(np.abs(matched_filter)/fs)
plt.figure()
plt.plot(t, matched_filter, color="yellow")
plt.xlim(0, size)
plt.xlabel('Time')
plt.ylabel('Use matched filter')
plt.grid(True)
#plt.savefig("SOG.png", format="png", bbox_inches="tight")
plt.show()

# Optimal filter
optimal_filter = 2*np.fft.ifft(bpsk_fft * bpsk_fft.conjugate()/power_vec)

# Returns complex conjugation - .conjugate()
#SNR_optimal = np.sqrt(np.abs(optimal_filter)/fs)
plt.figure()
plt.plot(t, optimal_filter, color="yellow")
plt.xlim(0, size)
plt.xlabel('Time')
plt.ylabel('Using the best filter')
plt.grid(True)
#plt.savefig("OP.png", format="png", bbox_inches="tight")
plt.show()
print(str([b11,a11]))
print(str([b12,a12]))
