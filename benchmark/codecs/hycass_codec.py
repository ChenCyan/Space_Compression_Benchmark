"""Learned-codec wrapper for all HyCASS-family models.

Supports:
  - HyCASS (AdjustableSpatioSpectral..., Swin Transformer + Conv)
  - CAE1D   (1D convolutional autoencoder, spectral only)
  - CAE3D   (3D convolutional autoencoder, joint spatio-spectral)
  - SSCNet  (Spectral Signals Compressor Network)
  - HYCOT   (HyCASS Transformer variant)

All models are from the same codebase (MIT License) and share:
  model.compress(x) → latent
  model.decompress(latent) → x_hat
  model.compression_ratio / model.bpppc
"""

from __future__ import annotations

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.domain import Domain, DEFAULT_DOMAIN


class HyCASSCodec(Codec):
    family = "lossy"

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        *,
        src_channels: int = 111,
        img_size=(80, 80),
        device: str | None = None,
        domain: Domain = DEFAULT_DOMAIN,
    ):
        super().__init__(domain)
        import torch, math, re
        from models import models as MODEL_REGISTRY

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dev = torch.device(device)
        self.device = "gpu" if self._dev.type == "cuda" else "cpu"

        # ---- HyCASS-specific: parse name, detect window_size, construct manually ----
        is_hycass = model_name.startswith("hycass_")
        if is_hycass:
            from models.hycass import AdjustableSpatioSpectralHyperspectralImageCompressionNetwork

            m = re.match(r"hycass_cr(\d+)_spatial(\d+)x_n(\d+)", model_name)
            if not m:
                raise ValueError(f"Cannot parse HyCASS model_name: {model_name!r}")
            cr_target = int(m.group(1))
            stages_spatial = int(m.group(2))
            N = int(m.group(3))

            window_size = 8
            ckpt_L = None
            ckpt_state = None
            if checkpoint_path is not None:
                ckpt_state = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(ckpt_state, dict) and "state_dict" in ckpt_state:
                    ckpt_state = ckpt_state["state_dict"]
                for key, val in ckpt_state.items():
                    if "relative_position_bias_table" in key:
                        n_entries = val.shape[0]
                        window_size = (int(math.sqrt(n_entries)) + 1) // 2
                    if key == "encoder.3.weight":
                        ckpt_L = val.shape[0]

            compression_ratio_spatial = 4 ** stages_spatial
            if ckpt_L is not None:
                cr_target = src_channels * compression_ratio_spatial / ckpt_L

            model = AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(
                src_channels=src_channels,
                img_size=img_size,
                cr_target=cr_target,
                stages_spatial=stages_spatial,
                N=N,
                window_size=window_size,
            )
        else:
            # ---- Non-HyCASS: use factory function from MODEL_REGISTRY ----
            model = MODEL_REGISTRY[model_name](src_channels=src_channels)
            ckpt_state = None
            if checkpoint_path is not None:
                ckpt_state = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(ckpt_state, dict) and "state_dict" in ckpt_state:
                    ckpt_state = ckpt_state["state_dict"]

        # ---- Load weights ----
        if ckpt_state is not None:
            model.load_state_dict(ckpt_state, strict=False)

        self.model = model.to(self._dev).eval()
        self.model_name = model_name
        self.arch_cr = float(getattr(model, "compression_ratio", float("nan")))
        self.bpppc = float(getattr(model, "bpppc", float("nan")))
        self.name = model_name  # e.g. "cae1d_cr004" or "hycass_cr222_spatial2x_n128"
        self.domain = Domain(bit_depth=domain.bit_depth, dn_scale=domain.dn_scale, bitdepth_native=32)

    def operating_point(self) -> str:
        return f"archCR={self.arch_cr:.1f}"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        import os, struct
        torch = self._torch
        _profile = os.environ.get("ENABLE_HYCASS_PROFILE") == "1"
        x_float = self.domain.to_float(x_dn)
        x = torch.from_numpy(x_float).unsqueeze(0).to(self._dev)
        with torch.no_grad():
            if _profile and self._dev.type == "cuda":
                # Per-module GPU timing via CUDA events.
                # encoder Sequential layout:
                #   [0] Conv2d(src_channels, N, 1)
                #   [1] LeakyReLU
                #   [2] Spatial Encoder (Sequential of RSTB+Conv stages; empty if stages_spatial=0)
                #   [3] Conv2d(N, L, 1)
                #   [4] Sigmoid
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                times_ms = {}
                encoder = self.model.encoder

                # enc_spectral: [0] + [1]
                ev_start.record()
                h = encoder[1](encoder[0](x))
                ev_end.record(); torch.cuda.synchronize()
                times_ms["enc_spectral"] = ev_start.elapsed_time(ev_end)

                # enc_spatial: [2]
                ev_start.record()
                h = encoder[2](h)
                ev_end.record(); torch.cuda.synchronize()
                times_ms["enc_spatial"] = ev_start.elapsed_time(ev_end)

                # enc_cr_adapter: [3] + [4]
                ev_start.record()
                latent = encoder[4](encoder[3](h))
                ev_end.record(); torch.cuda.synchronize()
                times_ms["enc_cr_adapter"] = ev_start.elapsed_time(ev_end)
            else:
                latent = self.model.compress(x)

        latent_np = latent.detach().to("cpu").numpy().astype(np.float32)
        meta = {
            "shape": tuple(x_dn.shape),
            "latent_shape": tuple(latent_np.shape),
            "arch_cr": self.arch_cr,
            "bpppc": self.bpppc,
        }
        if _profile and self._dev.type == "cuda":
            meta["profile_enc"] = times_ms
        # Fast serialization: raw bytes + shape (avoids np.save overhead).
        payload = struct.pack("<HH", len(meta["latent_shape"]), 0) + np.int32(meta["latent_shape"]).tobytes() + latent_np.tobytes()
        return EncodeResult(payload=payload, meta=meta)

    def decode(self, enc: EncodeResult) -> np.ndarray:
        import os, struct
        torch = self._torch
        _profile = os.environ.get("ENABLE_HYCASS_PROFILE") == "1"
        ndim = struct.unpack_from("<H", enc.payload, 0)[0]
        off = 4
        shape = tuple(np.frombuffer(enc.payload, dtype=np.int32, count=ndim, offset=off))
        off += ndim * 4
        latent_np = np.frombuffer(enc.payload, dtype=np.float32, count=int(np.prod(shape)), offset=off).reshape(shape).copy()
        latent = torch.from_numpy(latent_np).to(self._dev)
        with torch.no_grad():
            if _profile and self._dev.type == "cuda":
                # decoder Sequential layout:
                #   [0] Conv2d(L, N, 1)
                #   [1] LeakyReLU
                #   [2] Spatial Decoder (Sequential of stages; empty if stages_spatial=0)
                #   [3] Conv2d(N, src_channels, 1)
                #   [4] Sigmoid
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                times_ms = {}
                decoder = self.model.decoder

                # dec_cr_adapter: [0] + [1]
                ev_start.record()
                h = decoder[1](decoder[0](latent))
                ev_end.record(); torch.cuda.synchronize()
                times_ms["dec_cr_adapter"] = ev_start.elapsed_time(ev_end)

                # dec_spatial: [2]
                ev_start.record()
                h = decoder[2](h)
                ev_end.record(); torch.cuda.synchronize()
                times_ms["dec_spatial"] = ev_start.elapsed_time(ev_end)

                # dec_spectral: [3] + [4]
                ev_start.record()
                x_hat = decoder[4](decoder[3](h))
                ev_end.record(); torch.cuda.synchronize()
                times_ms["dec_spectral"] = ev_start.elapsed_time(ev_end)
            else:
                x_hat = self.model.decompress(latent)
        x_hat = x_hat.squeeze(0).detach().to("cpu").numpy()
        result = self.domain.to_dn(x_hat)
        if _profile and self._dev.type == "cuda":
            # Merge encode profile (stored in meta) with decode profile
            enc_times = enc.meta.get("profile_enc", {})
            all_times = {**enc_times, **times_ms}
            # Print profile summary to stderr so it doesn't contaminate CSV output
            import sys as _sys
            stages = self.model.stages_spatial
            enc_total = sum(enc_times.values())
            enc_pct = {k: v / max(enc_total, 0.01) * 100 for k, v in enc_times.items()}
            print(f"\n[HyCASS profile] {self.name}  stages_spatial={stages}  C={x_hat.shape[0]} H={x_hat.shape[1]} W={x_hat.shape[2]}",
                  file=_sys.stderr)
            print(f"  Encode ({enc_total:.3f} ms): "
                  f"spectral={enc_times['enc_spectral']:.3f}ms ({enc_pct['enc_spectral']:.0f}%)  "
                  f"spatial={enc_times['enc_spatial']:.3f}ms ({enc_pct['enc_spatial']:.0f}%)  "
                  f"cr_adapter={enc_times['enc_cr_adapter']:.3f}ms ({enc_pct['enc_cr_adapter']:.0f}%)",
                  file=_sys.stderr)
            print(f"  Decode: "
                  f"cr_adapter={times_ms['dec_cr_adapter']:.3f}ms  "
                  f"spatial={times_ms['dec_spatial']:.3f}ms  "
                  f"spectral={times_ms['dec_spectral']:.3f}ms",
                  file=_sys.stderr)
        return result
