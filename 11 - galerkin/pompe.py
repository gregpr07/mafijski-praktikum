# =============================================================================
# Galerkin method
# Authod: Miha Pompe
# =============================================================================

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy import integrate
import time
from matplotlib import cm
#import pandas as pd
from scipy import optimize
C_ref = 0.7577220257467273  # 0.757722123

# =============================================================================
# Core functions
# =============================================================================


def psi_mn(hi, fi, m, n):
    return hi**(2*m+1)*(1-hi)**n*np.sin((2*m+1)*fi)


def A_m_n_mn(m_, n_, m, n):
    if m_ != m:
        return 0
    return -np.pi/2*(n*n_*(3+4*m))/(2+4*m+n+n_)*sc.beta(n+n_-1, 3+4*m)


def b_m_n_(m_, n_):
    return -2/(2*m_+1)*sc.beta(2*m_+3, n_+1)


def to_cartesian(hi, fi, psi):
    r, angle = np.meshgrid(hi, fi)
    return [r*np.cos(angle), r*np.sin(angle), psi]


def compute_velocity(M, N):
    A = np.zeros((N*M, N*M))
    b = np.zeros(N*M)
    for m in range(M):
        for n in range(N):
            for n_ in range(N):
                A[m*N+n, m*N+n_] = A_m_n_mn(m, n_+1, m, n+1)
            b[m*N+n] = b_m_n_(m, n+1)
    a = np.linalg.solve(A, b)
    C = -32/np.pi * np.dot(b, a)
    return A, a, b, C


def analyse_error():
    C_ref = 0.7577220257467273  # 0.757722123
    Ns = np.arange(5, 35)
    Ms = np.arange(5, 35)
    C_error = np.zeros((30, 30))
    times = np.zeros((30, 30))
    for ni, N in enumerate(Ns):
        print(N)
        for mi, M in enumerate(Ms):
            time1 = time.time()
            _, _, _, C = compute_velocity(M, N)
            dt = time.time()-time1
            C_error[ni, mi] = np.abs(C-C_ref)
            times[ni, mi] = dt
    # np.save("data/C_error.npy", C_error)
    # np.save("data/times.npy", times)
    N_, M_ = np.meshgrid(Ns, Ms)
    return C_error, times, N_, M_


# =============================================================================
# Exercise functions
# =============================================================================

def plot_field():
    M = 20
    N = 20
    A, a, b, C = compute_velocity(M, N)
    print(f"C = {C}")

    N_hi = 500
    N_fi = 500
    hi = np.linspace(0, 1, N_hi)
    fi = np.linspace(0, np.pi, N_fi)
    psi = np.zeros((N_fi, N_hi))
    for m in range(M):
        for n in range(N):
            for i, fi_i in enumerate(fi):
                psi[i] += psi_mn(hi, fi_i, m, n+1)*a[m*N+n]
    out = to_cartesian(hi, fi, psi)
    plt.scatter(out[0], out[1], s=1, c=psi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Velocity profile, C = {C}")
    plt.colorbar()
    # plt.savefig("graphs/velocity_field.pdf")
    plt.show()


def time_and_error():
    C_ref = 0.757722123
    Ns = np.arange(5, 35)
    Ms = np.arange(5, 35)
    Ns, Ms = np.meshgrid(Ns, Ms)
    C_error = np.load("data/C_error.npy")
    times = np.load("data/times.npy")
    fig = plt.figure()

    # ax = plt.axes(projection='3d')
    # ax.view_init(15, 40)
    # ax.plot_surface(Ns, Ms, np.log(C_error), rstride=1, cstride=1, cmap='winter', edgecolor="black")
    # ax.set_ylabel("M")
    # ax.set_xlabel("N")
    # ax.set_zlabel(r"$log|C-C_{ref}|$")
    # ax.set_title("Error in C")
    # plt.savefig("graphs/error_in_c.pdf")
    # plt.show()

    ax = plt.axes(projection='3d')
    ax.view_init(13, -140)
    ax.plot_surface(Ns, Ms, times, rstride=1, cstride=1,
                    cmap='winter', edgecolor="black")
    ax.set_ylabel("M")
    ax.set_xlabel("N")
    ax.set_zlabel("t [s]")
    ax.set_title("Computation time")
    # plt.savefig("graphs/time.pdf")
    plt.show()


def base_functions():
    N_hi = 500
    N_fi = 500
    hi = np.linspace(0, 1, N_hi)
    fi = np.linspace(0, np.pi, N_fi)
    psi = np.zeros((N_fi, N_hi))
    cnt = 1
    plt.figure(figsize=(4, 7))
    for m in [0, 2, 4]:
        for n in [1]:
            for i, fi_i in enumerate(fi):
                psi[i] = psi_mn(hi, fi_i, m, n)
            print(np.min(psi))
            out = to_cartesian(hi, fi, psi)
            plt.subplot(3, 1, cnt)
            plt.title(f"m = {m}, n = {n}")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.scatter(out[0], out[1], c=psi, s=1)
            plt.colorbar()
            cnt += 1
    # plt.savefig("graphs/base_functions.pdf")
    plt.show()


def extra_exercise():
    N = 150
    Nx = 200
    Nt = 200
    t_ = np.linspace(0, 5, Nt)
    x_ = np.linspace(-2, 2, Nx)
    x, t = np.meshgrid(x_, t_)
    j = np.arange(-N//2, N//2)
    a_j = np.zeros_like(x, dtype=np.complex128)
    error = []
    a_j2 = np.sin(np.pi*np.cos(x+t))
    for j_ in j:
        a_j_zero2, _ = integrate.quad(lambda x: np.sin(
            np.pi*np.cos(x))*np.exp(-1j*j_*x), 0, 2*np.pi)/np.sqrt(2*np.pi)
        a_j_zero = np.sin(j_*np.pi/2)*sc.jv(j_, np.pi)
        error.append(np.abs(a_j_zero-a_j_zero2))
        a_j += np.exp(1j*j_*(x+t))/np.sqrt(2*np.pi)*a_j_zero2

    plt.figure(figsize=(12, 10))

    plt.subplot(224)
    plt.plot(j[N//2:], error[N//2:])
    plt.yscale("log")
    plt.xlabel(r"$j$")
    plt.ylabel(r"$a_j(0)$")
    plt.title(r"Absolute error in $a_j(0)$")

    plt.subplot(221)
    plt.title(r"Solution $u(x, t)$, $N=150$, top view")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.scatter(x, t, c=a_j.real, s=5)
    plt.colorbar()

    plt.subplot(222)
    plt.title(r"Solution $u(x, t)$, $N=150$")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    for i in range(0, 100):
        plt.plot(x_, a_j[:, i].real, c=cm.winter(i/100))
    plt.colorbar(cm.ScalarMappable(norm=cm.colors.Normalize(
        0, t_[100]), cmap=cm.winter), label=r"$t$", location="right")

    plt.subplot(223)
    plt.title("Absolute error")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.scatter(x, t, c=np.abs(a_j2.real-a_j.real), s=5)
    plt.colorbar()
    # plt.savefig("graphs/dodatna_solution2.pdf")
    plt.show()


if __name__ == "__main__":
    # extra_exercise()
    base_functions()
    time_and_error()
