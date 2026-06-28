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
# Plain-Python encoder/decoder (no numba, but pre-allocated bytearray)
# ============================================================================

def fast_encoder(mapped, D=16, U_MAX=8, GAMMA_0=1, GAMMA_STAR=5, K_ZPRIME=0):
    """Encode mapped deltas to bytes using pre-allocated bytearray."""

# ============================================================================
# Numba-jitted encoder (all logic inlined, no helper functions)
# ============================================================================

@njit
def _encode_numba(mapped, D, U_MAX, GAMMA_0, GAMMA_STAR, K_ZPRIME):
    Nx, Ny, Nz = mapped.shape
    est = int(Nx * Ny * Nz * 20 // 8 * 2)
    buf = np.zeros(est, dtype=np.uint8)
    pos = 0
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                t = y * Nx + x
                v = mapped[x, y, z]
                if t == 0:
                    gam = 1 << GAMMA_0
                    eps = int(np.floor((1./128.) * ((3*(1<<(K_ZPRIME+6)))-49) * gam))
                    for i in range(D-1, -1, -1):
                        if (v >> i) & 1:
                            buf[pos // 8] |= (1 << (7 - (pos % 8)))
                        pos += 1
                    continue
                thr = eps + int(np.floor((49./128.) * gam))
                kz = 0
                if 2*gam <= thr:
                    for i in range(D,0,-1):
                        if gam * (1<<i) <= thr:
                            kz = i
                            break
                q = v >> kz
                if q < U_MAX:
                    for _ in range(q): pos += 1  # q zeros
                    buf[pos // 8] |= (1 << (7 - (pos % 8))); pos += 1  # stop 1
                    if kz > 0:
                        r = v & ((1 << kz) - 1)
                        for i in range(kz-1, -1, -1):
                            if (r >> i) & 1:
                                buf[pos // 8] |= (1 << (7 - (pos % 8)))
                            pos += 1
                else:
                    for _ in range(U_MAX): pos += 1
                    for i in range(D-1, -1, -1):
                        if (v >> i) & 1:
                            buf[pos // 8] |= (1 << (7 - (pos % 8)))
                        pos += 1
                if gam < (1<<GAMMA_STAR)-1:
                    eps += v; gam += 1
                elif gam == (1<<GAMMA_STAR)-1:
                    eps = int(np.floor((eps + v + 1)/2))
                    gam = int(np.floor((gam + 1)/2))
    return buf[:(pos+7)//8], pos

# ============================================================================
# Numba-jitted decoder (all logic inlined)
# ============================================================================

@njit
def _decode_numba(bitdata, num_bits, Nx, Ny, Nz, D, U_MAX, GAMMA_0, GAMMA_STAR, K_ZPRIME):
    dec = np.zeros((Nx, Ny, Nz), dtype=np.int64)
    npix = Nx * Ny * Nz
    buf = bitdata
    bi = 0
    x, y, z = 0, 0, 0
    cnt = 0
    while cnt < npix and bi < num_bits:
        t = y * Nx + x
        if t == 0:
            gam = 1 << GAMMA_0
            eps = int(np.floor((1./128.) * ((3*(1<<(K_ZPRIME+6)))-49) * gam))
            vv = 0
            for i in range(D):
                bi2 = bi + i
                if (buf[bi2//8] >> (7-(bi2%8))) & 1:
                    vv |= (1 << (D-1-i))
            bi += D
            dec[x,y,z] = vv
            if x < Nx-1: x+=1
            elif y < Ny-1: x=0; y+=1
            else: x=0; y=0; z+=1
            cnt += 1
            continue
        thr = eps + int(np.floor((49./128.) * gam))
        kz = 0
        if 2*gam <= thr:
            for i in range(D,0,-1):
                if gam*(1<<i) <= thr:
                    kz = i
                    break
        q = 0
        while q < U_MAX and bi < num_bits:
            bi2 = bi
            if ((buf[bi2//8] >> (7-(bi2%8))) & 1) == 0:
                q += 1; bi += 1
            else:
                break
        if q == U_MAX:
            vv = 0
            for i in range(D):
                bi2 = bi + i
                if (buf[bi2//8] >> (7-(bi2%8))) & 1:
                    vv |= (1 << (D-1-i))
            bi += D
            dec[x,y,z] = vv
        else:
            bi += 1  # skip stop bit
            if kz == 0:
                dec[x,y,z] = q
            else:
                rr = 0
                for i in range(kz):
                    bi2 = bi + i
                    if (buf[bi2//8] >> (7-(bi2%8))) & 1:
                        rr |= (1 << (kz-1-i))
                bi += kz
                dec[x,y,z] = q*(1<<kz) + rr
        vv = dec[x,y,z]
        if gam < (1<<GAMMA_STAR)-1:
            eps += vv; gam += 1
        elif gam == (1<<GAMMA_STAR)-1:
            eps = int(np.floor((eps+vv+1)/2))
            gam = int(np.floor((gam+1)/2))
        if x < Nx-1: x+=1
        elif y < Ny-1: x=0; y+=1
        else: x=0; y=0; z+=1
        cnt += 1
    return dec

def fast_encoder(mapped, D=16, U_MAX=8, GAMMA_0=1, GAMMA_STAR=5, K_ZPRIME=0):
    buf, nbits = _encode_numba(mapped, D, U_MAX, GAMMA_0, GAMMA_STAR, K_ZPRIME)
    return buf.tobytes(), nbits

def fast_decoder(bitdata, num_bits, Nx, Ny, Nz, D=16, U_MAX=8, GAMMA_0=1, GAMMA_STAR=5, K_ZPRIME=0):
    buf = np.frombuffer(bitdata, dtype=np.uint8)
    return _decode_numba(buf, num_bits, Nx, Ny, Nz, D, U_MAX, GAMMA_0, GAMMA_STAR, K_ZPRIME)
