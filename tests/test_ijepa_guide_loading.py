import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.guides.ijepa import IJEPAGuide
from src.models.vision_transformer import VisionTransformer


class IJEPAGuideLoadingTests(unittest.TestCase):
    def test_loads_extracted_safetensors_state(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "encoder.safetensors"
            model = VisionTransformer(
                img_size=32,
                patch_size=16,
                embed_dim=24,
                depth=1,
                num_heads=3,
            )
            save_file(
                {key: value.detach().contiguous() for key, value in model.state_dict().items()},
                str(path),
            )
            guide = IJEPAGuide(
                weights_path=str(path),
                weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input_size=32,
                patch_size=16,
                embed_dim=24,
                depth=1,
                num_heads=3,
                device="cpu",
                dtype="float32",
            )
            output = guide.encode(torch.rand(2, 3, 32, 32))
            self.assertEqual(tuple(output.patch_tokens.shape), (2, 4, 24))
            self.assertEqual(output.metadata["weights_format"], "safetensors")
            self.assertTrue(output.metadata["weights_sha256_verified"])
            self.assertIsNone(output.native_map)
            guide.cleanup()

    def test_rejects_weights_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "encoder.safetensors"
            path.write_bytes(b"not the expected checkpoint")
            with self.assertRaises(RuntimeError):
                IJEPAGuide(
                    weights_path=str(path),
                    weights_sha256="0" * 64,
                    input_size=32,
                    patch_size=16,
                    embed_dim=24,
                    depth=1,
                    num_heads=3,
                    device="cpu",
                    dtype="float32",
                )


if __name__ == "__main__":
    unittest.main()
