import json

import numpy as np
import pytest
import torch

from src import eval_downstream as evaluation


def test_complete_checkpoint_digest_not_shared_prefix(tmp_path):
    first, second = tmp_path / 'a.pt', tmp_path / 'b.pt'
    first.write_bytes(b'x' * (1 << 20) + b'A')
    second.write_bytes(b'x' * (1 << 20) + b'B')
    assert evaluation.file_sha256(first) != evaluation.file_sha256(second)


def _manifest():
    return {'subject_ids': ['a', 'b'], 'num_slices': 3, 'source': 'checkpoint-a'}


def test_cache_provenance_and_legacy_are_explicit():
    manifest = _manifest()
    data = {'features': torch.zeros(2, 3, 8), 'labels': torch.tensor([0, 1]),
            'source_manifest': manifest}
    evaluation._validate_feature_cache(data, manifest)
    with pytest.raises(ValueError, match="identity"):
        evaluation._validate_feature_cache(data, dict(manifest, source='checkpoint-b'))
    del data['source_manifest']
    with pytest.raises(ValueError, match="identity"):
        evaluation._validate_feature_cache(data, manifest)
    evaluation._validate_feature_cache(data, manifest, allow_unverified=True)
    data['labels'] = torch.tensor([0])
    with pytest.raises(ValueError, match="length"):
        evaluation._validate_feature_cache(data, manifest, allow_unverified=True)


def test_prediction_order_and_manifest_are_exported(tmp_path):
    path = tmp_path / 'predictions.npz'
    evaluation.save_predictions(str(path), np.array([0, 1]), np.array([0.2, 0.7]), _manifest())
    with np.load(path) as data:
        np.testing.assert_array_equal(data['subject_ids'], ['a', 'b'])
        np.testing.assert_array_equal(data['row_index'], [0, 1])
        assert data['source_manifest_sha256'].item() == evaluation._identity_digest(_manifest())
    assert json.loads(path.with_suffix('.manifest.json').read_text()) == _manifest()
    with pytest.raises(ValueError, match="length"):
        evaluation.save_predictions(str(path), [0], [0.2], _manifest())


def test_real_tiny_feature_cache_roundtrip_and_precision(tmp_path, monkeypatch):
    from torch import nn
    training = tmp_path / 'Training'
    training.mkdir()
    for i in range(2):
        np.savez(training / ('case%d.npz' % i),
                 oct_bscans=np.full((200, 4, 4), 30 + i * 100, dtype=np.uint8),
                 glaucoma=np.array(i))

    class TinyEncoder(nn.Module):
        def forward(self, images):
            return images.mean(dim=(2, 3))[:, None, :]

    encoder = TinyEncoder()
    evaluation.set_amp(True)  # Per-call fp32 must override the global setting.
    kwargs = dict(encoder=encoder, data_dir=str(tmp_path), split='Training',
                  num_slices=2, slice_size=4, device='cpu', chunk_size=1,
                  cache_dir=str(tmp_path / 'cache'), use_amp=False,
                  return_manifest=True, num_workers=0)
    features, labels, manifest = evaluation.precompute_features(**kwargs)
    assert features.dtype == torch.float32 and manifest['use_amp'] is False
    assert labels.tolist() == [0, 1]
    assert manifest['ordered_files'][0]['name'] == 'case0.npz'
    monkeypatch.setattr(encoder, 'forward', lambda x: pytest.fail("Cache should avoid encoding"))
    cached, _, again = evaluation.precompute_features(**kwargs)
    torch.testing.assert_close(features, cached)
    assert again == manifest
