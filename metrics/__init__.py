from metrics import mse
from metrics import psnr
from metrics import sa

metrics = {
    "mse": mse.MeanSquaredError,
    "psnr": psnr.PeakSignalToNoiseRatio,
    "sa": sa.SpectralAngle,
}
