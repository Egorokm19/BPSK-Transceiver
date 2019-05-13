# BPSK-Transceiver
Implementing a transmitter and receiver model of a BPSK signal in Python.
The model consists of:
1. Transmitter models:
1) source of pseudo-random binary data
2) forming filter
3) quadrature modulation in the transmitter with the addition of carrier frequency mismatch of approximately Fs /10,000 and symbol frequency mismatch
2. Receiver models:
1) with matched filter
2) character sync
3) carrier synchronization
3. Models of communication channel (the simplest - AWGN)
Mismatch parameters and models (frequencies, bit rates, etc.) are determined by variables that are initialized at the very beginning of the simulation.
Results for the project:
![Иллюстрация к проекту](https://github.com/Egorokm19/BPSK-Transceiver/edit/master/image/S1.png)
