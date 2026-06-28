"""Optimized CCSDS-123 encoder/decoder using bytearray + numba predictor.

The original encoder/decoder used Python list with .append() for every bit
(~9M operations), which was the dominant bottleneck (19+16=35s per sample).

This rewrite uses:
  - Pre-allocated bytearray with direct bit indexing (no dynamic list growth)
  - Inlined dec_to_bin / bin_to_dec avoiding string conversion
  - Numba-accelerated predictor/unpredictor
"""

import numpy as np
from numba import njit

# ============================================================================
# Numba-accelerated predictor / unpredictor helpers
# ============================================================================

@njit
def _sign(x):
    return 1 if x >= 0 else -1


@njit
def _local_sum(x, y, z, data):
    Nx = data.shape[0]
    if y > 0 and x > 0 and x < (Nx - 1):
        return data[x-1, y, z] + data[x-1, y-1, z] + data[x, y-1, z] + data[x+1, y-1, z]
    if y == 0 and x > 0:
        return 4 * data[x-1, y, z]
    if y > 0 and x == 0:
        return 2 * (data[x, y-1, z] + data[x+1, y-1, z])
    if y > 0 and x == (Nx - 1):
        return data[x-1, y, z] + data[x-1, y-1, z] + 2 * data[x, y-1, z]
    return 0


@njit
def _local_diff_vector(x, y, z, data, P):
    Nz = data.shape[2]
    pz = min(P, z)
    size = 3 + pz
    ld = np.zeros(size, dtype=np.float64)
    if y > 0:
        ld[0] = 4 * data[x, y-1, z] - _local_sum(x, y, z, data)
    if x > 0 and y > 0:
        ld[1] = 4 * data[x-1, y, z] - _local_sum(x, y, z, data)
        ld[2] = 4 * data[x-1, y-1, z] - _local_sum(x, y, z, data)
    elif x == 0 and y > 0:
        ld[1] = 4 * data[x, y-1, z] - _local_sum(x, y, z, data)
        ld[2] = 4 * data[x, y-1, z] - _local_sum(x, y, z, data)
    for i in range(1, pz + 1):
        ld[2 + i] = 4 * data[x, y, z-i] - _local_sum(x, y, z-i, data)
    return ld


@njit
def _init_weight_vector(z, P, W_RES):
    pz = min(P, z)
    size = 3 + pz
    wv = np.zeros(size, dtype=np.float64)
    for i in range(1, pz + 1):
        if i == 1:
            wv[2 + i] = (7.0 / 8.0) * (2 ** W_RES)
        else:
            wv[2 + i] = np.floor((1.0 / 8.0) * wv[2 + i - 1])
    return wv


@njit
def _prediction(ld_vector, weight_vector, x, y, z, data, S_MIN, S_MAX, S_MID, W_RES, R, P):
    d = np.sum(weight_vector * ld_vector)
    temp1 = d + (2 ** W_RES) * (_local_sum(x, y, z, data) - 4 * S_MID)
    temp2 = (temp1 + (2 ** (R - 1))) % (2 ** R) - 2 ** (R - 1)
    temp3 = temp2 + (2 ** (W_RES + 2) * S_MID) + 2 ** (W_RES + 1)
    min_val = 2 ** (W_RES + 2) * S_MIN
    max_val = 2 ** (W_RES + 2) * S_MAX + 2 ** (W_RES + 1)
    hr_s = min(max(temp3, min_val), max_val)
    Nx = data.shape[0]
    t = y * Nx + x
    s_tilde = 0.0
    if t > 0:
        s_tilde = np.floor(hr_s / (2 ** (W_RES + 1)))
    elif t == 0 and P > 0 and z > 0:
        s_tilde = 2 * data[x, y, z - 1]
    elif t == 0 and (P == 0 or z == 0):
        s_tilde = 2 * S_MID
    s_hat = np.floor(s_tilde / 2)
    return s_hat, s_tilde


@njit
def _updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data, V_MIN, V_MAX, T_INC, D, W_RES, W_MIN, W_MAX):
    Nx = data.shape[0]
    t = y * Nx + x
    dr_e = 2 * data[x, y, z] - s_tilde
    vv = min(max(V_MIN + np.floor((t - Nx) / T_INC), V_MIN), V_MAX)
    w_exp = vv + D - W_RES
    base = _sign(dr_e) * (2 ** (-w_exp))
    n = len(weight_vector)
    new_wv = np.zeros(n, dtype=np.float64)
    for i in range(n):
        new_wv[i] = min(max(weight_vector[i] + np.floor(0.5 * (base * ld_vector[i])), W_MIN), W_MAX)
    return new_wv


@njit
def _mapper(delta, s_hat, s_tilde, S_MIN, S_MAX):
    theta = min(s_hat - S_MIN, S_MAX - s_hat)
    if abs(delta) > theta:
        return abs(delta) + theta
    sig = 1 if s_tilde % 2 == 0 else -1
    if 0 <= sig * delta and sig * delta <= theta:
        return 2 * abs(delta)
    else:
        return 2 * abs(delta) - 1


@njit
def _unmapper(mapped_delta, s_hat, s_tilde, S_MIN, S_MAX):
    theta = min(s_hat - S_MIN, S_MAX - s_hat)
    if mapped_delta - theta > theta:
        if theta == s_hat - S_MIN:
            return mapped_delta - theta
        else:
            return theta - mapped_delta
    elif mapped_delta % 2 == 0:
        if s_tilde % 2 == 0:
            return mapped_delta / 2
        else:
            return -(mapped_delta / 2)
    else:
        if (s_tilde + 1) % 2 == 0:
            return (mapped_delta + 1) / 2
        else:
            return -((mapped_delta + 1) / 2)


@njit
def _predictor_numba(data, D, P, S_MIN, S_MAX, S_MID, W_RES, W_MIN, W_MAX, R, V_MIN, V_MAX, T_INC):
    Nx, Ny, Nz = data.shape
    mapped = np.empty_like(data)
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                t = y * Nx + x
                if t == 0:
                    weight_vector = _init_weight_vector(z, P, W_RES)
                ld_vector = _local_diff_vector(x, y, z, data, P)
                s_hat, s_tilde = _prediction(ld_vector, weight_vector, x, y, z, data, S_MIN, S_MAX, S_MID, W_RES, R, P)
                weight_vector = _updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data, V_MIN, V_MAX, T_INC, D, W_RES, W_MIN, W_MAX)
                mapped[x, y, z] = _mapper(data[x, y, z] - s_hat, s_hat, s_tilde, S_MIN, S_MAX)
    return mapped


@njit
def _unpredictor_numba(mapped, D, P, S_MIN, S_MAX, S_MID, W_RES, W_MIN, W_MAX, R, V_MIN, V_MAX, T_INC):
    Nx, Ny, Nz = mapped.shape
    data = np.zeros_like(mapped)
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                t = y * Nx + x
                if t == 0:
                    weight_vector = _init_weight_vector(z, P, W_RES)
                ld_vector = _local_diff_vector(x, y, z, data, P)
                s_hat, s_tilde = _prediction(ld_vector, weight_vector, x, y, z, data, S_MIN, S_MAX, S_MID, W_RES, R, P)
                delta = _unmapper(mapped[x, y, z], s_hat, s_tilde, S_MIN, S_MAX)
                data[x, y, z] = s_hat + delta
                weight_vector = _updated_weight_vector(s_tilde, weight_vector, ld_vector, x, y, z, data, V_MIN, V_MAX, T_INC, D, W_RES, W_MIN, W_MAX)
    return data


# ============================================================================
# Fast encoder/decoder (pre-allocated bytearray, no Python list)
# ============================================================================

def _make_params(D=16, P=2, W_RES=4, R=32):
    """Return a dict of CCSDS-123 parameters for the encoder/decoder."""
    S_MIN = -(2 ** (D - 1))
    S_MAX = 2 ** (D - 1)
    S_MID = 0
    K = 0
    GAMMA_STAR = 5
    U_MAX = 8
    GAMMA_0 = 1
    K_ZPRIME = K if K <= 30 - D else 2 * K + D - 30
    return dict(D=D, P=P, W_RES=W_RES, R=R, S_MIN=S_MIN, S_MAX=S_MAX, S_MID=S_MID,
                K=K, GAMMA_STAR=GAMMA_STAR, U_MAX=U_MAX, GAMMA_0=GAMMA_0, K_ZPRIME=K_ZPRIME)


def fast_predictor(data, D=16, P=2, W_RES=4, R=32):
    p = _make_params(D, P, W_RES, R)
    return _predictor_numba(data, D, P, p["S_MIN"], p["S_MAX"], p["S_MID"],
                            W_RES, -(2**(W_RES+1)), 2**(W_RES+2)-1, R, -6, 9, 2**4)


def fast_unpredictor(mapped, D=16, P=2, W_RES=4, R=32):
    p = _make_params(D, P, W_RES, R)
    return _unpredictor_numba(mapped, D, P, p["S_MIN"], p["S_MAX"], p["S_MID"],
                              W_RES, -(2**(W_RES+1)), 2**(W_RES+2)-1, R, -6, 9, 2**4)


# ============================================================================
# Optimized encoder (bytearray — NO python list append)
# ============================================================================

def _dec_to_bin_list(val, width):
    """Return list of bits (0/1 ints) for val, width bits, MSB first."""
    return [(val >> (width - 1 - i)) & 1 for i in range(width)]


def fast_encoder(mapped, D=16, U_MAX=8, GAMMA_0=1, GAMMA_STAR=5, K_ZPRIME=0):
    """Encode mapped deltas to a bytes object directly (no intermediate list).

    Uses bytearray + bit position tracking instead of list.append().
    This avoids 9M Python append() calls and 9M list element allocations.
    """
    Nx, Ny, Nz = mapped.shape
    # Pre-allocate generous buffer (typical bitrate ~13 bpppc for lossless).
    n_pixels = Nx * Ny * Nz
    est_bytes = int(n_pixels * 16 / 8 * 1.5)
    buf = bytearray(est_bytes)
    bit_pos = 0  # next write position in bits

    def write_bits(bits):
        nonlocal bit_pos
        for b in bits:
            if bit_pos // 8 >= len(buf):
                buf.extend(b'\x00' * 1024)
            if b:
                buf[bit_pos // 8] |= (1 << (7 - (bit_pos % 8)))
            bit_pos += 1

    def write_value(val, nbits):
        nonlocal bit_pos
        for i in range(nbits - 1, -1, -1):
            b = (val >> i) & 1
            if bit_pos // 8 >= len(buf):
                buf.extend(b'\x00' * 1024)
            if b:
                buf[bit_pos // 8] |= (1 << (7 - (bit_pos % 8)))
            bit_pos += 1

    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                t = y * Nx + x
                val = int(mapped[x, y, z])

                if t == 0:
                    gamma = 1 << GAMMA_0
                    epsilon_z = int(np.floor((1.0 / 128.0) * ((3 * (1 << (K_ZPRIME + 6))) - 49) * gamma))
                    write_value(val, D)
                    continue

                # compute k_z
                thr = epsilon_z + int(np.floor((49.0 / 128.0) * gamma))
                if 2 * gamma > thr:
                    k_z = 0
                else:
                    k_z = 0
                    for i in range(D, 0, -1):
                        if gamma * (1 << i) <= thr:
                            k_z = i
                            break

                # GPO2 encode
                q = val >> k_z
                if q < U_MAX:
                    # unary part: q zeros + 1
                    write_bits([0] * q + [1])
                    if k_z > 0:
                        write_value(val & ((1 << k_z) - 1), k_z)
                else:
                    write_bits([0] * U_MAX)
                    write_value(val, D)

                # update state
                if gamma < (1 << GAMMA_STAR) - 1:
                    epsilon_z += val
                    gamma += 1
                elif gamma == (1 << GAMMA_STAR) - 1:
                    epsilon_z = int(np.floor((epsilon_z + val + 1) / 2))
                    gamma = int(np.floor((gamma + 1) / 2))

    # Return bytes
    return buf[: (bit_pos + 7) // 8], bit_pos


# ============================================================================
# Optimized decoder (bytearray — NO python list)
# ============================================================================

def fast_decoder(bitdata, num_bits, Nx, Ny, Nz, D=16, U_MAX=8, GAMMA_0=1, GAMMA_STAR=5, K_ZPRIME=0):
    """Decode bytes back to mapped_delta array (Nx,Ny,Nz)."""
    decoded = np.zeros((Nx, Ny, Nz), dtype=np.int64)
    n_pixels = Nx * Ny * Nz
    buf = bytearray(bitdata)

    def read_bit(pos):
        return (buf[pos // 8] >> (7 - (pos % 8))) & 1

    def read_value(pos, nbits):
        v = 0
        for i in range(nbits):
            if read_bit(pos + i):
                v |= (1 << (nbits - 1 - i))
        return v, pos + nbits

    bit_i = 0
    x, y, z = 0, 0, 0
    pixel_count = 0

    while pixel_count < n_pixels:
        t = y * Nx + x

        if t == 0:
            gamma = 1 << GAMMA_0
            epsilon_z = int(np.floor((1.0 / 128.0) * ((3 * (1 << (K_ZPRIME + 6))) - 49) * gamma))
            val, bit_i = read_value(bit_i, D)
            decoded[x, y, z] = val
            # advance
            if x < Nx - 1: x += 1
            elif y < Ny - 1: x = 0; y += 1
            elif z < Nz - 1: x = 0; y = 0; z += 1
            pixel_count += 1
            continue

        thr = epsilon_z + int(np.floor((49.0 / 128.0) * gamma))
        if 2 * gamma > thr:
            k_z = 0
        else:
            k_z = 0
            for i in range(D, 0, -1):
                if gamma * (1 << i) <= thr:
                    k_z = i
                    break

        # GPO2 decode
        q = 0
        while q < U_MAX and bit_i < num_bits and read_bit(bit_i) == 0:
            q += 1
            bit_i += 1

        if q == U_MAX:
            val, bit_i = read_value(bit_i, D)
        else:
            bit_i += 1  # skip the terminating 1
            if k_z == 0:
                val = q
            else:
                r, bit_i = read_value(bit_i, k_z)
                val = q * (1 << k_z) + r

        decoded[x, y, z] = val

        # update state
        if gamma < (1 << GAMMA_STAR) - 1:
            epsilon_z += val
            gamma += 1
        elif gamma == (1 << GAMMA_STAR) - 1:
            epsilon_z = int(np.floor((epsilon_z + val + 1) / 2))
            gamma = int(np.floor((gamma + 1) / 2))

        # advance
        if x < Nx - 1: x += 1
        elif y < Ny - 1: x = 0; y += 1
        elif z < Nz - 1: x = 0; y = 0; z += 1
        pixel_count += 1

    return decoded
