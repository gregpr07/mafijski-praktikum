# import time
# from tqdm import tqdm
# import math
# from scipy import special, stats, optimize, fft
# import matplotlib.pyplot as plt
# import numpy as np

# from multiprocessing import Pool


# def Hn(n, N, delta):
#     return np.sum(np.exp(-2j * np.pi * n * np.arange(N-1) / (N-1)) * x) * delta


# def fourier_basic(x, delta):
#     N = len(x) + 1
#     coeffs = np.empty((N-1, N-1))
#     # Hns = np.empty(N-1)

#     def Hnn(n): return Hn(n, N, delta)

#     with Pool(5) as p:
#         print(p.map(Hnn, range(N-1)))
#     # for n in range(N-1):
#     #     Hns[n] = Hn(n)
#         # return Hns
#     # return np.dot(coeffs, x)*delta


# frequencies = [882, 1378, 2756, 5512, 11025, 44100]

# for freq in tqdm(frequencies[:2]):
#     sig = np.loadtxt(f'./bach/Bach.{freq}.txt')
#     t0 = 0
#     t1 = 2.3
#     rate = freq*2

#     N = len(sig)

#     delta = 1/rate
#     freqs = np.linspace(0.0, int(rate/2), int(N/2))
#     x = np.linspace(t0, t1, num=N)

#     freqs_y = fourier_basic(sig, delta)
#     # ker itak dobimo simetricno
#     y = 2/N * np.abs(freqs_y[0:int(N/2)])

#     # plt.plot(x, sig)
#     plt.plot(freqs[:2000], y[:2000], label=fr'$\nu_c = {freq}Hz$')

# plt.legend()
# plt.show()
