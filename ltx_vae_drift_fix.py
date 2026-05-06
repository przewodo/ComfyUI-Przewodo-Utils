import torch
import torch.nn.functional as F


class LTXVAEDriftFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "decoded": ("IMAGE",),
                "mode": ([
                    "covariance_global",
                    "percentile_global",
                    "mean_std_global",
                    "affine_rgb_global",
                ], {"default": "covariance_global"}),

                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),

                "low_percentile": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),

                "high_percentile": ("FLOAT", {
                    "default": 99.9,
                    "min": 90.0,
                    "max": 100.0,
                    "step": 0.1,
                }),

                "max_samples": ("INT", {
                    "default": 1000000,
                    "min": 10000,
                    "max": 3000000,
                    "step": 10000,
                }),

                "calibration_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected",)
    FUNCTION = "fix"
    CATEGORY = "LTX/Color"

    def _match_batch(self, img, target_b):
        if img.shape[0] == target_b:
            return img

        if img.shape[0] == 1:
            return img.repeat(target_b, 1, 1, 1)

        reps = (target_b + img.shape[0] - 1) // img.shape[0]
        return img.repeat(reps, 1, 1, 1)[:target_b]

    def _resize_to_square(self, img, size):
        img_nchw = img.permute(0, 3, 1, 2)
        img_resized = F.interpolate(
            img_nchw,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        return img_resized.permute(0, 2, 3, 1)

    def _safe_sample_flat(self, img, max_samples):
        flat = img.reshape(-1, 3)
        total = flat.shape[0]

        if total <= max_samples:
            return flat

        idx = torch.randint(
            low=0,
            high=total,
            size=(max_samples,),
            device=img.device,
            dtype=torch.long,
        )

        return flat[idx]

    def _safe_sample_pair(self, original, decoded, max_samples):
        original_flat = original.reshape(-1, 3)
        decoded_flat = decoded.reshape(-1, 3)

        total = min(original_flat.shape[0], decoded_flat.shape[0])

        if total <= max_samples:
            return original_flat[:total], decoded_flat[:total]

        idx = torch.randint(
            low=0,
            high=total,
            size=(max_samples,),
            device=decoded.device,
            dtype=torch.long,
        )

        return original_flat[idx], decoded_flat[idx]

    def _matrix_sqrt_inv(self, mat, eps=1e-5):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = eigvals.clamp_min(eps)
        inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T
        return inv_sqrt

    def _matrix_sqrt(self, mat, eps=1e-5):
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = eigvals.clamp_min(eps)
        sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        return sqrt

    def _covariance_global(self, original, decoded, max_samples, calibration_size):
        """
        Unpaired global color distribution matching.

        It matches decoded to original using:
        - RGB mean
        - RGB covariance

        This is better than simple RGB affine when the image is not perfectly
        aligned or when the VAE changes contrast/saturation globally.
        """

        original = self._match_batch(original, decoded.shape[0])

        original_calib = self._resize_to_square(original, calibration_size)
        decoded_calib = self._resize_to_square(decoded, calibration_size)

        original_sample = self._safe_sample_flat(original_calib, max_samples).float()
        decoded_sample = self._safe_sample_flat(decoded_calib, max_samples).float()

        orig_mean = original_sample.mean(dim=0, keepdim=True)
        dec_mean = decoded_sample.mean(dim=0, keepdim=True)

        orig_centered = original_sample - orig_mean
        dec_centered = decoded_sample - dec_mean

        n_o = max(original_sample.shape[0] - 1, 1)
        n_d = max(decoded_sample.shape[0] - 1, 1)

        orig_cov = (orig_centered.T @ orig_centered) / n_o
        dec_cov = (dec_centered.T @ dec_centered) / n_d

        eye = torch.eye(3, device=decoded.device, dtype=torch.float32)
        orig_cov = orig_cov + eye * 1e-5
        dec_cov = dec_cov + eye * 1e-5

        dec_inv_sqrt = self._matrix_sqrt_inv(dec_cov)
        orig_sqrt = self._matrix_sqrt(orig_cov)

        transform = dec_inv_sqrt @ orig_sqrt

        decoded_flat = decoded.reshape(-1, 3).float()
        corrected = (decoded_flat - dec_mean) @ transform + orig_mean

        return corrected.reshape_as(decoded)

    def _affine_rgb_global(self, original, decoded, max_samples, calibration_size):
        original = self._match_batch(original, decoded.shape[0])

        original_calib = self._resize_to_square(original, calibration_size)
        decoded_calib = self._resize_to_square(decoded, calibration_size)

        y, x = self._safe_sample_pair(
            original_calib,
            decoded_calib,
            max_samples,
        )

        x = x.float()
        y = y.float()

        ones = torch.ones(
            (x.shape[0], 1),
            device=x.device,
            dtype=x.dtype,
        )

        x_aug = torch.cat([x, ones], dim=1)

        xtx = x_aug.T @ x_aug
        xty = x_aug.T @ y

        ridge = 1e-4
        eye = torch.eye(4, device=x.device, dtype=x.dtype)
        eye[3, 3] = 0.0

        matrix = torch.linalg.solve(xtx + ridge * eye, xty)

        decoded_flat = decoded.reshape(-1, 3).float()

        decoded_aug = torch.cat([
            decoded_flat,
            torch.ones(
                (decoded_flat.shape[0], 1),
                device=decoded.device,
                dtype=decoded_flat.dtype,
            )
        ], dim=1)

        corrected = decoded_aug @ matrix

        return corrected.reshape_as(decoded)

    def _mean_std_global(self, original, decoded, max_samples, eps=1e-6):
        original_sample = self._safe_sample_flat(original, max_samples).float()
        decoded_sample = self._safe_sample_flat(decoded, max_samples).float()

        original_mean = original_sample.mean(dim=0).view(1, 1, 1, 3)
        decoded_mean = decoded_sample.mean(dim=0).view(1, 1, 1, 3)

        original_std = original_sample.std(dim=0).clamp_min(eps).view(1, 1, 1, 3)
        decoded_std = decoded_sample.std(dim=0).clamp_min(eps).view(1, 1, 1, 3)

        corrected = (decoded - decoded_mean) / decoded_std * original_std + original_mean

        return corrected

    def _percentile_global(
        self,
        original,
        decoded,
        low_percentile,
        high_percentile,
        max_samples,
        eps=1e-6,
    ):
        q_low = low_percentile / 100.0
        q_high = high_percentile / 100.0

        original_sample = self._safe_sample_flat(original, max_samples).float()
        decoded_sample = self._safe_sample_flat(decoded, max_samples).float()

        original_low = torch.quantile(original_sample, q_low, dim=0).view(1, 1, 1, 3)
        original_high = torch.quantile(original_sample, q_high, dim=0).view(1, 1, 1, 3)

        decoded_low = torch.quantile(decoded_sample, q_low, dim=0).view(1, 1, 1, 3)
        decoded_high = torch.quantile(decoded_sample, q_high, dim=0).view(1, 1, 1, 3)

        original_range = (original_high - original_low).clamp_min(eps)
        decoded_range = (decoded_high - decoded_low).clamp_min(eps)

        corrected = (decoded - decoded_low) / decoded_range * original_range + original_low

        return corrected

    def fix(
        self,
        original,
        decoded,
        mode,
        strength,
        low_percentile,
        high_percentile,
        max_samples,
        calibration_size,
    ):
        original_dtype = decoded.dtype

        original = original.float().clamp(0.0, 1.0)
        decoded = decoded.float().clamp(0.0, 1.0)

        original = self._match_batch(original, decoded.shape[0])

        if mode == "covariance_global":
            corrected = self._covariance_global(
                original,
                decoded,
                max_samples,
                calibration_size,
            )

        elif mode == "affine_rgb_global":
            corrected = self._affine_rgb_global(
                original,
                decoded,
                max_samples,
                calibration_size,
            )

        elif mode == "percentile_global":
            corrected = self._percentile_global(
                original,
                decoded,
                low_percentile,
                high_percentile,
                max_samples,
            )

        elif mode == "mean_std_global":
            corrected = self._mean_std_global(
                original,
                decoded,
                max_samples,
            )

        else:
            corrected = decoded

        corrected = corrected.clamp(0.0, 1.0)

        output = decoded * (1.0 - strength) + corrected * strength
        output = output.clamp(0.0, 1.0).to(original_dtype)

        return (output,)