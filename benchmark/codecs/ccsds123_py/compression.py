"""CCSDS 123.0-B-2 compression module (from BrianShTsoi/CCSDS-Compression-Python).

This is a verbatim copy of the reference implementation, placed here as a
local module so the Codec wrapper can import it via importlib without the
global-variable collisions we hit with the NTNU model.

Original: https://github.com/BrianShTsoi/CCSDS-Compression-Python
"""

import numpy as np

# Parameters for Compression
D = 16  # dynamic range, number of bits for sample size value (between 2 to 32)
S_MIN = -1 * (2 ** (D - 1))
S_MAX = 2 ** (D - 1)
S_MID = 0

P = 2  # number of previous bands used for prediction (between 2-15)
W_RES = 4  # Omega, between 4 and 19
W_MIN = -(2 ** (W_RES + 1))
W_MAX = 2 ** (W_RES + 2) - 1
R = 32
M = 0  # max error (0 = lossless)

V_MIN = -6
V_MAX = 9
T_INC = 2 ** 4

# Parameters for Decompression
U_MAX = 8
GAMMA_0 = 1

K = 0
GAMMA_STAR = 5

if K <= 30 - D:
    K_ZPRIME = K
else:
    K_ZPRIME = 2 * K + D - 30


def sign(x):
    return 1 if x >= 0 else -1


def dec_to_bin(d, width):
    if isinstance(d, np.float64):
        d = int(d)
    b = [int(bit) for bit in bin(d)[2:]]
    if len(b) <= width:
        b = [0] * abs(len(b) - width) + b
    else:
        b = b[-width:]
    return b


def bin_to_dec(b):
    d = 0
    for bit in b:
        d = (d << 1) | bit
    return d


def local_sum(x, y, z, data):
    Nx = data.shape[0]
    if y > 0 and x > 0 and x < (Nx - 1):
        return data[x - 1, y, z] + data[x - 1, y - 1, z] + data[x, y - 1, z] + data[x + 1, y - 1, z]
    elif y == 0 and x > 0:
        return 4 * (data[x - 1, y, z])
    elif y > 0 and x == 0:
        return 2 * (data[x, y - 1, z] + data[x + 1, y - 1, z])
    elif y > 0 and x == (Nx - 1):
        return data[x - 1, y, z] + data[x - 1, y - 1, z] + 2 * data[x, y - 1, z]
    return 0


def Ps(z):
    return min(P, z)


def local_diff_vector(x, y, z, data):
    ld_vector = []
    dN, dW, dNW = 0, 0, 0
    if y > 0:
        dN = 4 * (data[x, y - 1, z]) - local_sum(x, y, z, data)
    elif y == 0:
        dN = 0
    if x > 0 and y > 0:
        dW = 4 * (data[x - 1, y, z]) - local_sum(x, y, z, data)
        dNW = 4 * (data[x - 1, y - 1, z]) - local_sum(x, y, z, data)
    elif x == 0 and y > 0:
        dW = 4 * (data[x, y - 1, z]) - local_sum(x, y, z, data)
        dNW = 4 * (data[x, y - 1, z]) - local_sum(x, y, z, data)
    elif y == 0:
        dW = 0
        dNW = 0
    ld_vector.extend([dN, dW, dNW])
    for i in range(1, Ps(z) + 1):
        d = 4 * (data[x, y, z - i]) - local_sum(x, y, z - i, data)
        ld_vector.append(d)
    return np.array(ld_vector)


def init_weight_vector(z):
    init_weight_vector = []
    wN, wW, wNW = 0, 0, 0
    init_weight_vector.extend([wN, wW, wNW])
    for i in range(1, Ps(z) + 1):
        if i == 1:
            w1 = (7 / 8) * (2 ** W_RES)
            init_weight_vector.append(w1)
        else:
            wi = np.floor((1 / 8) * init_weight_vector[-1])
            init_weight_vector.append(wi)
    return np.array(init_weight_vector)


def prediction(ld_vector, weight_vector, x, y, z, data):
    d = np.inner(weight_vector, ld_vector)
    temp1 = d + (2 ** W_RES) * (local_sum(x, y, z, data) - 4 * S_MID)
    temp2 = (temp1 + (2 ** (R - 1))) % (2 ** R) - 2 ** (R - 1)
    temp3 = temp2 + (2 ** (W_RES + 2) * S_MID) + 2 ** (W_RES + 1)
    min_val = 2 ** (W_RES + 2) * S_MIN
    max_val = 2 ** (W_RES + 2) * S_MAX + 2 ** (W_RES + 1)
    hr_s = np.clip(temp3, min_val, max_val)
    Nx = data.shape[0]
    t = y * Nx + x
    s_tilde = 0
    if t > 0:
        s_tilde = np.floor(hr_s / (2 ** (W_RES + 1)))
    elif t == 0 and P > 0 and z > 0:
        s_tilde = 2 * data[x, y, z - 1]
    elif t == 0 and (P == 0 or z == 0):
        s_tilde = 2 * S_MID
    s_hat = np.floor(s_tilde / 2)
    return s_hat, s_tilde


def updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data):
    Nx = data.shape[0]
    t = y * Nx + x
    new_weight_vector = []
    dr_e = 2 * data[x, y, z] - s_tilde
    w_exp = np.clip(V_MIN + np.floor((t - Nx) / T_INC), V_MIN, V_MAX) + D - W_RES
    base = (sign(dr_e)) * (2 ** (-(w_exp)))
    for i in range(len(weight_vector)):
        new_w = np.clip(weight_vector[i] + np.floor((1 / 2) * (base * ld_vector[i])), W_MIN, W_MAX)
        new_weight_vector.append(new_w)
    return np.array(new_weight_vector)


def mapper(delta, s_hat, s_tilde):
    theta = min(s_hat - S_MIN, S_MAX - s_hat)
    if abs(delta) > theta:
        return abs(delta) + theta
    elif 0 <= (-1) ** int(s_tilde) * delta and (-1) ** int(s_tilde) * delta <= theta:
        return 2 * abs(delta)
    else:
        return 2 * abs(delta) - 1


def predictor(data):
    Nx = data.shape[0]
    Ny = data.shape[1]
    Nz = data.shape[2]
    mapped = np.empty_like(data)
    for z in range(0, Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                t = y * Nx + x
                if t == 0:
                    weight_vector = init_weight_vector(z)
                ld_vector = local_diff_vector(x, y, z, data)
                s_hat, s_tilde = prediction(ld_vector, weight_vector, x, y, z, data)
                weight_vector = updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data)
                mapped[x, y, z] = mapper(data[x, y, z] - s_hat, s_hat, s_tilde)
    return mapped


def GPO2(k, j):
    if (np.floor(j / (2 ** k)) < U_MAX):
        u = [0] * int(np.floor(j / (2 ** k))) + [1]
        r = dec_to_bin(j, D)[-k:] if k != 0 else []
    else:
        u = [0] * U_MAX
        r = dec_to_bin(j, D)
    return u + r


def encoder(mapped_delta):
    Nx = mapped_delta.shape[0]
    Ny = mapped_delta.shape[1]
    Nz = mapped_delta.shape[2]
    encoded = []
    for z in range(0, Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                t = y * Nx + x
                if t == 0:
                    gamma = 2 ** GAMMA_0
                    epsilon_z = np.floor((1 / (2 ** 7)) * ((3 * (2 ** (K_ZPRIME + 6))) - 49) * gamma)
                    codeword = dec_to_bin(mapped_delta[x, y, z], D)
                    encoded += codeword
                    continue
                if 2 * gamma > epsilon_z + np.floor((49 / (2 ** 7)) * gamma):
                    k_z = 0
                else:
                    for i in range(D, 0, -1):
                        if gamma * (2 ** i) <= epsilon_z + np.floor((49 / (2 ** 7)) * gamma):
                            k_z = i
                            break
                encoded += GPO2(k_z, mapped_delta[x, y, z])
                if gamma < 2 ** GAMMA_STAR - 1:
                    epsilon_z = epsilon_z + mapped_delta[x, y, z]
                    gamma = gamma + 1
                elif gamma == 2 ** GAMMA_STAR - 1:
                    epsilon_z = np.floor((epsilon_z + mapped_delta[x, y, z] + 1) / 2)
                    gamma = np.floor((gamma + 1) / 2)
    return encoded


def increment_xyz(x, y, z, Nx, Ny, Nz):
    if x < Nx - 1:
        return x + 1, y, z
    elif y < Ny - 1:
        return 0, y + 1, z
    elif z < Nz - 1:
        return 0, 0, z + 1
    else:
        return None


def inv_GPO2(k, encoded, i):
    j = 0
    q = 0
    while i < len(encoded) and encoded[i] == 0 and q < U_MAX:
        i += 1
        q += 1
    if q == U_MAX:
        j = bin_to_dec(encoded[i:i + D])
        i += D
    else:
        i += 1
        r = 0 if k == 0 else bin_to_dec(encoded[i:i + k])
        j = q * 2 ** k + r
        i += k
    return j, i


def decoder(encoded, Nx, Ny, Nz):
    decoded = np.zeros((Nx, Ny, Nz), dtype=np.int64)
    x, y, z = 0, 0, 0
    i = 0
    while i < len(encoded):
        t = y * Nx + x
        if t == 0:
            gamma = 2 ** GAMMA_0
            epsilon_z = np.floor((1 / (2 ** 7)) * ((3 * (2 ** (K_ZPRIME + 6))) - 49) * gamma)
            codeword = encoded[i:i + D]
            i += D
            value = bin_to_dec(codeword)
            decoded[x, y, z] = value
            x, y, z = increment_xyz(x, y, z, Nx, Ny, Nz)
            continue
        if 2 * gamma > epsilon_z + np.floor((49 / (2 ** 7)) * gamma):
            k_z = 0
        else:
            for idx in range(D, 0, -1):
                if gamma * (2 ** idx) <= epsilon_z + np.floor((49 / (2 ** 7)) * gamma):
                    k_z = idx
                    break
        value, i = inv_GPO2(k_z, encoded, i)
        decoded[x, y, z] = value
        if x < Nx - 1 or y < Ny - 1 or z < Nz - 1:
            x, y, z = increment_xyz(x, y, z, Nx, Ny, Nz)
        if gamma < 2 ** GAMMA_STAR - 1:
            epsilon_z = epsilon_z + value
            gamma = gamma + 1
        elif gamma == 2 ** GAMMA_STAR - 1:
            epsilon_z = np.floor((epsilon_z + value + 1) / 2)
            gamma = np.floor((gamma + 1) / 2)
    return decoded


def unmapper(mapped_delta, s_hat, s_tilde):
    theta = min(s_hat - S_MIN, S_MAX - s_hat)
    if mapped_delta - theta > theta:
        return mapped_delta - theta if theta == s_hat - S_MIN else theta - mapped_delta
    elif mapped_delta % 2 == 0:
        return (-1) ** int(s_tilde) * mapped_delta / 2
    else:
        return (-1) ** (int(s_tilde) + 1) * (mapped_delta + 1) / 2


def unpredictor(mapped, Nx, Ny, Nz):
    data = np.zeros_like(mapped)
    for z in range(0, Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                t = y * Nx + x
                if t == 0:
                    weight_vector = init_weight_vector(z)
                ld_vector = local_diff_vector(x, y, z, data)
                s_hat, s_tilde = prediction(ld_vector, weight_vector, x, y, z, data)
                delta = unmapper(mapped[x, y, z], s_hat, s_tilde)
                data[x, y, z] = s_hat + delta
                weight_vector = updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data)
    return data
