"""Minimal witness of the audited cache-iterator/default-RNG coupling."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.eval_downstream import LinearHead


def draw_head(cache_miss, local_probe_seed):
    torch.manual_seed(42)
    data = TensorDataset(torch.zeros(2, 1))
    if cache_miss:
        for _ in range(3):
            # Each feature split's iterator consumes a default-RNG base seed.
            iterator = iter(DataLoader(data, batch_size=1, shuffle=False, num_workers=0))
            del iterator
    if local_probe_seed:
        torch.manual_seed(42)
    head = LinearHead(768)
    flat = torch.cat([parameter.detach().flatten() for parameter in head.parameters()])
    return flat


def main():
    out = Path(__file__).with_suffix(".json")
    if out.exists():
        raise FileExistsError(out)
    warm = draw_head(False, False)
    cold = draw_head(True, False)
    reseeded_warm = draw_head(False, True)
    reseeded_cold = draw_head(True, True)
    if torch.equal(warm, cold):
        raise AssertionError("Witness failed to reproduce default-RNG coupling")
    if not torch.equal(reseeded_warm, reseeded_cold):
        raise AssertionError("Local probe seeding failed to remove the isolated coupling")
    result = {
        "scope": "Minimal source-pattern witness using the real LinearHead and DataLoader iterator, not an end-to-end historical fit",
        "cached_features_differ": False,
        "warm_cold_initial_parameter_max_difference": float((warm - cold).abs().max()),
        "with_local_reseed_parameter_max_difference": float((reseeded_warm - reseeded_cold).abs().max()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_code_changed": False,
        "limits": [
            "Frozen encoder construction also consumes RNG; held common and omitted in this minimal witness.",
            "Does not identify the historical initial states or quantify an AUC effect.",
            "Production fix should isolate head initialization and training-loader RNG; this script changes neither.",
        ],
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
