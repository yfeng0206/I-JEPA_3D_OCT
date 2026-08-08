#!/usr/bin/env python3
"""
Measure informative context content in I-JEPA masks.
Compares random rectangle vs anatomy-guided target masks over 1000 slices.
"""
import sys
import pathlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# Setup paths
TRAIN = pathlib.Path(r'D:\jepa_phase0\fairvision-glaucoma\data\Training')
GRIDS_PATH = pathlib.Path(r'D:\jepa_phase0\mirage-goals\outputs\budget\grids_1k.npz')
OUTPUT_DIR = pathlib.Path('results/masking/diagnostics')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, 'src')

def load_grids():
    """Load pre-computed MIRAGE grids for anatomy targets."""
    with np.load(GRIDS_PATH) as z:
        per = z['per']  # (1000, 2, 16, 16) - [P_inner, P_choroid]
    return per

def reproduce_image_crops(n_slices=1000):
    """
    Reproduce the EXACT crop loop from the description.
    Returns list of (jimg_256, raw_image) tuples.
    """
    vols = sorted(p.stem for p in TRAIN.glob('data_*.npz'))
    rng = np.random.default_rng(7)
    N = n_slices
    per_vol = max(1, N // max(len(vols), 1))
    
    images = []
    for vi, v in enumerate(vols):
        if len(images) >= N:
            break
        try:
            with np.load(TRAIN/(v+'.npz'), allow_pickle=True) as z:
                vol = z['oct_bscans']
        except Exception as e:
            print(f"Failed to load {v}: {e}")
            continue
        
        for d in rng.choice(len(vol), size=min(per_vol, len(vol)), replace=False):
            raw = np.asarray(vol[int(d)], np.float32)
            lo, hi = raw.min(), raw.max()
            u = (raw-lo)/(hi-lo) if hi > lo else raw*0
            pil = Image.fromarray((u*255).astype(np.uint8))

            # RandomResizedCrop.get_params draws from the GLOBAL TORCH rng, not
            # from `rng` above (which only picks the slice).  Without this seed
            # the crop geometry is fresh randomness on every run, so the image
            # is not even reproducible against itself.
            torch.manual_seed(1000 + len(images))
            # Get the same RandomResizedCrop params as original
            i, j, h, w = T.RandomResizedCrop.get_params(pil, scale=(0.3,1.0), ratio=(3/4,4/3))
            
            # JEPA 256x256 crop
            jimg = np.asarray(TF.resized_crop(pil, i, j, h, w, [256,256], T.InterpolationMode.BICUBIC), np.float32)/255.
            
            images.append(jimg)
            if len(images) >= N:
                break
    
    print(f"Reproduced {len(images)} crops (expected {N})")
    return np.array(images)  # (1000, 256, 256)

class MaskCollatorSimple:
    """Simplified MaskCollator for random rectangle generation."""
    def __init__(self, input_size=(256,256), patch_size=16):
        self.input_size = input_size
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size  # 16
        self.width = input_size[1] // patch_size   # 16
        self.npred = 4
        self.pred_mask_scale = (0.15, 0.2)
        # Must match src/masks/multiblock.py.  This was (0.4, 0.5) and produced
        # a context block SMALLER than the reported context, which is
        # impossible; prefer ctx_anatomy_probe.py, which drives the real
        # collator instead of reimplementing it.
        self.enc_mask_scale = (0.85, 1.0)
    
    def _sample_block_size(self, scale, g):
        """Sample block size in patches."""
        lo, hi = scale
        scale_val = lo + (hi - lo) * torch.rand(1, generator=g).item()
        area = (self.height * self.width) * scale_val
        w_patch = max(1, int(np.sqrt(area * 3/4)))
        h_patch = max(1, int(np.sqrt(area * 4/3)))
        w_patch = min(w_patch, self.width)
        h_patch = min(h_patch, self.height)
        return h_patch, w_patch
    
    def _sample_block_location(self, h, w, height, width, g=None):
        """Sample block top-left location."""
        if g is not None:
            top = torch.randint(0, height - h + 1, (1,), generator=g).item() if h < height else 0
            left = torch.randint(0, width - w + 1, (1,), generator=g).item() if w < width else 0
        else:
            top = np.random.randint(0, max(1, height - h + 1))
            left = np.random.randint(0, max(1, width - w + 1))
        return top, left
    
    def _block_to_indices(self, top, left, h, w):
        """Convert block (top, left, h, w) to flat grid indices."""
        indices = []
        for r in range(top, min(top + h, self.height)):
            for c in range(left, min(left + w, self.width)):
                indices.append(r * self.width + c)
        return indices
    
    def sample_targets_and_context(self, seed):
        """Sample 4 target blocks and 1 context block."""
        g = torch.Generator().manual_seed(seed)
        random.seed(seed)
        
        target_union = set()
        for _ in range(self.npred):
            bh, bw = self._sample_block_size(self.pred_mask_scale, g)
            top, left = self._sample_block_location(bh, bw, self.height, self.width, g)
            idx = self._block_to_indices(top, left, bh, bw)
            target_union.update(idx)
        
        # Context block
        bh, bw = self._sample_block_size(self.enc_mask_scale, g)
        top, left = self._sample_block_location(bh, bw, self.height, self.width, g)
        context_block = set(self._block_to_indices(top, left, bh, bw))
        
        context = context_block - target_union
        return target_union, context, context_block

    def context_for_union(self, seed, target_union):
        """The SAME context policy, applied to an externally supplied union.

        Without this the anatomy arm used `all 256 patches - targets`, which is
        defect D1 that anatomy_target_sampler_v2 was written to fix: it hands
        the encoder ~2x the image the rect arm gets, so the two arms have
        different denominators and any 'extra informative tokens' comparison
        between them is meaningless.  Replaying the target draws first keeps the
        RNG at the same position, so both arms see the identical context block.
        """
        g = torch.Generator().manual_seed(seed)
        random.seed(seed)
        for _ in range(self.npred):
            bh, bw = self._sample_block_size(self.pred_mask_scale, g)
            self._sample_block_location(bh, bw, self.height, self.width, g)

        bh, bw = self._sample_block_size(self.enc_mask_scale, g)
        top, left = self._sample_block_location(bh, bw, self.height, self.width, g)
        context_block = set(self._block_to_indices(top, left, bh, bw))
        return context_block - target_union, context_block

def build_anatomy_targets(P_inner, P_choroid):
    """Build anatomy-guided targets from MIRAGE grids."""
    # Only the IMPORT may fall back.  Wrapping the build_targets CALL as well
    # would swallow real sampler failures (it imports scipy lazily inside
    # fill_small_holes / n_components) and silently substitute a 128-cell
    # percentile mask that has nothing to do with the sampler.
    try:
        from anatomy_target_sampler_v2 import build_targets
    except ImportError:
        grid = np.maximum(P_inner, P_choroid)
        union = grid > np.percentile(grid, 50)
    else:
        parts, _regions = build_targets([P_inner, P_choroid], 4,
                                        mass_cap=0.80, tau=0.10, overlap=0.0)
        union = np.logical_or.reduce(parts)

    # np.argwhere returns (row, col) PAIRS; flattening them interleaves
    # coordinates and yields values in 0..15 instead of flat indices 0..255.
    return set(np.flatnonzero(union.ravel()).tolist())

def get_patch_stats(image_256x256, patch_size=16):
    """
    Compute mean intensity and variance for each patch.
    image: (256, 256) float in [0, 1]
    Returns: (256,) means, (256,) variances for flat grid indices
    """
    h_patches = w_patches = image_256x256.shape[0] // patch_size
    means = np.zeros(h_patches * w_patches)
    variances = np.zeros(h_patches * w_patches)
    
    for grid_idx in range(h_patches * w_patches):
        row = grid_idx // w_patches
        col = grid_idx % w_patches
        patch = image_256x256[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
        means[grid_idx] = patch.mean()
        variances[grid_idx] = patch.var()
    
    return means, variances

def count_informative(context_indices, means, variances, per_image_threshold_pctl=50):
    """
    Count informative tokens using three definitions.
    Returns counts for each definition and the actual threshold values used.
    """
    if not context_indices:
        return {'def_a': 0, 'def_b': 0, 'def_c': 0}, {}
    
    ctx_list = list(context_indices)
    ctx_means = means[ctx_list]
    ctx_vars = variances[ctx_list]
    
    # Def A: mean > 0.15
    def_a = np.sum(ctx_means > 0.15)
    
    # Def B: variance > median variance (computed over entire image)
    var_threshold = np.median(variances)
    def_b = np.sum(ctx_vars > var_threshold)
    
    # Def C: mean > per-image median patch intensity
    image_median_mean = np.median(means)
    def_c = np.sum(ctx_means > image_median_mean)
    
    return {
        'def_a': int(def_a),
        'def_b': int(def_b),
        'def_c': int(def_c)
    }, {
        'threshold_mean_0p15': 0.15,
        'threshold_var_median': float(var_threshold),
        'threshold_mean_image_median': float(image_median_mean),
        'ctx_mean_min': float(ctx_means.min()),
        'ctx_mean_max': float(ctx_means.max()),
        'ctx_var_min': float(ctx_vars.min()),
        'ctx_var_max': float(ctx_vars.max())
    }

def analyze_slice(image, grids_slice, seed, coll):
    """
    Analyze one slice for both methods.
    Returns: {rect: {...}, anatomy: {...}}
    """
    # Patch statistics for this image
    means, variances = get_patch_stats(image)
    
    # Random rectangle method
    target_rect, context_rect, _ = coll.sample_targets_and_context(seed)
    
    # Anatomy method -- SAME context policy as the rect arm, not all-256
    target_anat = build_anatomy_targets(grids_slice[0], grids_slice[1])
    context_anat, _ = coll.context_for_union(seed, target_anat)
    
    # Count informative for both
    info_rect, thresholds = count_informative(context_rect, means, variances)
    info_anat, _ = count_informative(context_anat, means, variances)
    
    # Information content: sum of patch variances
    total_var = variances.sum()
    context_rect_var = variances[list(context_rect)].sum() if context_rect else 0
    context_anat_var = variances[list(context_anat)].sum() if context_anat else 0
    
    total_mean = means.sum()
    context_rect_mean = means[list(context_rect)].sum() if context_rect else 0
    context_anat_mean = means[list(context_anat)].sum() if context_anat else 0
    
    return {
        'rect': {
            'total_tokens': len(context_rect),
            'informative': info_rect,
            'variance_retained': float(context_rect_var / total_var) if total_var > 0 else 0,
            'mean_retained': float(context_rect_mean / total_mean) if total_mean > 0 else 0,
        },
        'anatomy': {
            'total_tokens': len(context_anat),
            'informative': info_anat,
            'variance_retained': float(context_anat_var / total_var) if total_var > 0 else 0,
            'mean_retained': float(context_anat_mean / total_mean) if total_mean > 0 else 0,
        },
        'thresholds': thresholds
    }

def main():
    print("Loading grids...")
    grids = load_grids()  # (1000, 2, 16, 16)
    
    print("Reproducing image crops...")
    images = reproduce_image_crops(1000)  # (1000, 256, 256)
    
    print("Setting up mask collator...")
    coll = MaskCollatorSimple()
    
    # Accumulate results
    results_list = []
    all_thresholds = None
    
    print("Analyzing 1000 slices...")
    for idx in range(len(images)):
        if idx % 100 == 0:
            print(f"  Slice {idx}/1000")
        
        result = analyze_slice(images[idx], grids[idx], seed=idx, coll=coll)
        results_list.append(result)
        
        if all_thresholds is None:
            all_thresholds = result['thresholds']
    
    # Aggregate statistics
    rect_stats = {
        'total_tokens_mean': np.mean([r['rect']['total_tokens'] for r in results_list]),
        'total_tokens_std': np.std([r['rect']['total_tokens'] for r in results_list]),
        'informative_def_a_mean': np.mean([r['rect']['informative']['def_a'] for r in results_list]),
        'informative_def_b_mean': np.mean([r['rect']['informative']['def_b'] for r in results_list]),
        'informative_def_c_mean': np.mean([r['rect']['informative']['def_c'] for r in results_list]),
        'variance_retained_mean': np.mean([r['rect']['variance_retained'] for r in results_list]),
        'mean_retained_mean': np.mean([r['rect']['mean_retained'] for r in results_list]),
    }
    
    anat_stats = {
        'total_tokens_mean': np.mean([r['anatomy']['total_tokens'] for r in results_list]),
        'total_tokens_std': np.std([r['anatomy']['total_tokens'] for r in results_list]),
        'informative_def_a_mean': np.mean([r['anatomy']['informative']['def_a'] for r in results_list]),
        'informative_def_b_mean': np.mean([r['anatomy']['informative']['def_b'] for r in results_list]),
        'informative_def_c_mean': np.mean([r['anatomy']['informative']['def_c'] for r in results_list]),
        'variance_retained_mean': np.mean([r['anatomy']['variance_retained'] for r in results_list]),
        'mean_retained_mean': np.mean([r['anatomy']['mean_retained'] for r in results_list]),
    }
    
    # Compute fractions
    rect_stats['frac_informative_def_a'] = rect_stats['informative_def_a_mean'] / rect_stats['total_tokens_mean']
    rect_stats['frac_informative_def_b'] = rect_stats['informative_def_b_mean'] / rect_stats['total_tokens_mean']
    rect_stats['frac_informative_def_c'] = rect_stats['informative_def_c_mean'] / rect_stats['total_tokens_mean']
    
    anat_stats['frac_informative_def_a'] = anat_stats['informative_def_a_mean'] / anat_stats['total_tokens_mean']
    anat_stats['frac_informative_def_b'] = anat_stats['informative_def_b_mean'] / anat_stats['total_tokens_mean']
    anat_stats['frac_informative_def_c'] = anat_stats['informative_def_c_mean'] / anat_stats['total_tokens_mean']
    
    output = {
        'crops_note': 'Images reproduced using RNG seed=7 with RandomResizedCrop.get_params in same order',
        'grids_match': 'Positional match: image[i] paired with grids[i]',
        'thresholds': all_thresholds,
        'rect': rect_stats,
        'anatomy': anat_stats,
        'per_slice': results_list[:10]  # Store first 10 for inspection
    }
    
    # Save results
    with open(OUTPUT_DIR / 'ctx_informative.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("RESULTS: Informative Context Token Analysis")
    print("="*70)
    
    print(f"\nTHRESHOLDS USED:")
    print(f"  Def A: mean intensity > 0.15")
    print(f"  Def B: variance > {all_thresholds['threshold_var_median']:.6f} (median across dataset)")
    print(f"  Def C: mean intensity > per-image median = {all_thresholds['threshold_mean_image_median']:.4f} (average)")
    
    print(f"\nRANDOM RECTANGLES:")
    print(f"  Total context tokens: {rect_stats['total_tokens_mean']:.1f} ± {rect_stats['total_tokens_std']:.1f}")
    print(f"  Informative (Def A, mean>0.15): {rect_stats['informative_def_a_mean']:.1f} ({rect_stats['frac_informative_def_a']:.1%})")
    print(f"  Informative (Def B, var>med):   {rect_stats['informative_def_b_mean']:.1f} ({rect_stats['frac_informative_def_b']:.1%})")
    print(f"  Informative (Def C, mean>img):  {rect_stats['informative_def_c_mean']:.1f} ({rect_stats['frac_informative_def_c']:.1%})")
    print(f"  Variance retained: {rect_stats['variance_retained_mean']:.1%}")
    print(f"  Mean intensity retained: {rect_stats['mean_retained_mean']:.1%}")
    
    print(f"\nANATOMY-GUIDED:")
    print(f"  Total context tokens: {anat_stats['total_tokens_mean']:.1f} ± {anat_stats['total_tokens_std']:.1f}")
    print(f"  Informative (Def A, mean>0.15): {anat_stats['informative_def_a_mean']:.1f} ({anat_stats['frac_informative_def_a']:.1%})")
    print(f"  Informative (Def B, var>med):   {anat_stats['informative_def_b_mean']:.1f} ({anat_stats['frac_informative_def_b']:.1%})")
    print(f"  Informative (Def C, mean>img):  {anat_stats['informative_def_c_mean']:.1f} ({anat_stats['frac_informative_def_c']:.1%})")
    print(f"  Variance retained: {anat_stats['variance_retained_mean']:.1%}")
    print(f"  Mean intensity retained: {anat_stats['mean_retained_mean']:.1%}")
    
    print(f"\nCOMPARISON:")
    print(f"  Extra tokens in anatomy: {anat_stats['total_tokens_mean'] - rect_stats['total_tokens_mean']:.1f}")
    extra_def_a = anat_stats['informative_def_a_mean'] - rect_stats['informative_def_a_mean']
    extra_def_b = anat_stats['informative_def_b_mean'] - rect_stats['informative_def_b_mean']
    extra_def_c = anat_stats['informative_def_c_mean'] - rect_stats['informative_def_c_mean']
    print(f"  Extra informative tokens (Def A): {extra_def_a:.1f}")
    print(f"  Extra informative tokens (Def B): {extra_def_b:.1f}")
    print(f"  Extra informative tokens (Def C): {extra_def_c:.1f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {OUTPUT_DIR / 'ctx_informative.json'}")
    print("="*70)

if __name__ == '__main__':
    main()
