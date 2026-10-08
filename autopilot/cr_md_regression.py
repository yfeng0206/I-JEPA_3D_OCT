"""Pre-registered secondary analysis: visual-field mean deviation (MD) severity regression.

PREREGISTRATION_cr_seed_v1.md section 5: ridge regression of MD on the primary 768-d volume
representation (patch-token mean per B-scan, mean over B-scans) read from each probe's
feature cache; alpha selected on validation; MAE, RMSE and Spearman correlation.

- PRIMARY: glaucoma cases. Ridge fitted on glaucoma Training cases, alpha chosen on glaucoma
  Validation cases (minimum RMSE, ties -> larger alpha), evaluated on glaucoma Test cases.
- DESCRIPTIVE: full cohort, fitted/selected/evaluated on all cases of the same splits.
- Baselines: training-mean MD; class-mean MD diagnostic (true label -> training class mean;
  it uses the diagnosis label, so it is a reference for label-only information, not a
  competing model; inside the glaucoma subset it equals the training mean).
- Within each registered seed index: paired case bootstrap (same resampled cases for both
  encoders; label-stratified for the full cohort) of metric differences
  CENTROID / ENVELOPE / RANDOM-CB minus RANDOM. Anchors are reported per run only.
  This is the same cohort as the primary task, not an independent endpoint.

Seal and provenance (same pathway as test AUC; checked before any feature, name or MD value
is read):
- every run's results.json identity must equal the identity under its hash-verified seal
  manifest (manifest JSON only), and every cache must be a v2 cache of the requested split
  whose bytes match the recorded hash and whose manifest names the sealed checkpoint;
- runs whose caches lack verified per-case identity (legacy label-only caches, e.g. the
  original RANDOM ep50 anchor) are excluded from MD altogether and listed as excluded;
- within-seed contrasts use only runs that finished before the freeze (amendment 1);
- ``--split validation`` (default; allowed any time) never opens a Test cache, and a metadata
  row whose use is not Training/Validation is refused before its MD is converted. Alpha is
  chosen on the same split, so its metrics are optimistic;
- ``--split test`` refuses before the results freeze (2026-10-22 08:00 PDT); for each run the
  unseal receipt must exist and match the seal BEFORE any sealed array is opened, then all
  sealed hashes and the results/seal file coverage are re-verified.
  ``--allow-before-freeze`` is for tests only.
The alpha grid, tie rule, standardization, baselines and bootstrap recipe are this
module's implementation choices (fixed before any Test analysis), not text of section 5.

Light by design: CPU, <= 2 BLAS threads by default, caches read by memory map in chunks
(about 60 MB of features per run in memory).

Examples
  python autopilot/cr_md_regression.py --runs D:/jepa_phase0/runs/cr_probe_* --out md_val.json
  python autopilot/cr_md_regression.py --split test --runs ... --out md_test.json   # after freeze
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '2')

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_stats  # noqa: E402

METADATA = r'D:\jepa_phase0\fairvision-glaucoma\metadata\data_summary_glaucoma.csv'
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
SPLIT_USE = {'Training': 'training', 'Validation': 'validation', 'Test': 'test'}
CONTRAST_ARMS = ('centroid', 'envelope', 'random_cb')
N_BOOT = 10000
BOOT_SEED = 20261023


def _identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     default=str).encode('utf-8')).hexdigest()


def load_metadata(path, names, allowed_uses):
    """Parse only rows whose filename is in ``names``; a row whose ``use`` is not allowed
    (e.g. test in validation mode) is refused before its MD or diagnosis is converted."""
    wanted, out = set(names), {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader([stream.readline()]))
        col = {name: i for i, name in enumerate(header)}
        for line in stream:
            if line.split(',', 1)[0] not in wanted:
                continue
            row = next(csv.reader([line]))
            use = row[col['use']].strip().lower()
            if use not in allowed_uses:
                raise cr_stats.IntegrityError("metadata row %s has use=%s, not allowed here"
                                              % (row[col['filename']], use))
            out[row[col['filename']]] = {'md': float(row[col['md']]),
                                         'glaucoma': row[col['glaucoma']].strip().lower() == 'yes',
                                         'use': use}
    missing = wanted - set(out)
    if missing:
        raise cr_stats.IntegrityError("%d cases missing from the metadata, e.g. %s"
                                      % (len(missing), sorted(missing)[:3]))
    return out


def load_split(res, split, ident):
    """(features, labels, case names) of one split, bound to the sealed encoder identity.

    Before any feature or name is used: the cache must be a v2 cache of this split, its
    bytes must match the hash recorded at probe time, its manifest digest must match the
    probe record, and its encoder must be the checkpoint named under the seal. Legacy
    (label-only row identity) caches are never used for per-case MD regression.
    """
    import torch
    torch.set_num_threads(min(2, torch.get_num_threads()))
    run_dir = res['_run_dir']
    rec = res['cache_provenance'][split]['primary']
    path = rec.get('path') or ''
    if rec.get('state') not in ('cold', 'warm'):
        raise cr_stats.IntegrityError("%s: %s cache state %r has no verified per-case identity"
                                      % (run_dir, split, rec.get('state')))
    if not os.path.basename(path).startswith(split + '_'):
        raise cr_stats.IntegrityError("%s: %s cache path %s is not a %s cache"
                                      % (run_dir, split, path, split))
    if not rec.get('file_sha256') or cr_stats.sha256_file(path) != rec['file_sha256']:
        raise cr_stats.IntegrityError("%s: %s cache bytes differ from the recorded hash"
                                      % (run_dir, split))
    data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    manifest = data.get('source_manifest') or {}
    if manifest.get('split') != split:
        raise cr_stats.IntegrityError("%s: cache under %s holds split %r"
                                      % (run_dir, split, manifest.get('split')))
    if _identity_digest(manifest) != rec.get('identity_sha256'):
        raise cr_stats.IntegrityError("%s: %s cache identity differs from the probe record"
                                      % (run_dir, split))
    if (manifest.get('encoder_source') or {}).get('checkpoint_sha256') != \
            ident.get('checkpoint_sha256'):
        raise cr_stats.IntegrityError("%s: %s cache was not extracted from the sealed checkpoint"
                                      % (run_dir, split))
    feats = data['features']
    vol = np.concatenate([feats[i:i + 200].double().mean(dim=1).numpy()
                          for i in range(0, feats.shape[0], 200)])
    labels = data['labels'].numpy().astype(np.int64)
    names = [item['name'] for item in manifest['ordered_files']]
    if len(names) != len(labels):
        raise cr_stats.IntegrityError("%s: %s names/rows mismatch" % (run_dir, split))
    return vol, labels, names


def _attach_md(vol, labels, names, meta, split, run_dir):
    md = np.array([meta[n]['md'] for n in names], dtype=np.float64)
    glaucoma = np.array([meta[n]['glaucoma'] for n in names], dtype=bool)
    if not np.array_equal(glaucoma.astype(np.int64), labels):
        raise cr_stats.IntegrityError("%s: %s cache labels disagree with the metadata join"
                                      % (run_dir, split))
    if any(meta[n]['use'] != SPLIT_USE[split] for n in names):
        raise cr_stats.IntegrityError("%s: %s cases are not all metadata use=%s"
                                      % (run_dir, split, SPLIT_USE[split]))
    return {'x': vol, 'md': md, 'glaucoma': glaucoma, 'names': names}


class RidgePath(object):
    """Ridge on training-standardized features for a whole alpha grid (one SVD)."""

    def __init__(self, x, y):
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        self.y_mean = float(y.mean())
        u, s, vt = np.linalg.svd((x - self.mu) / self.sd, full_matrices=False)
        self.s, self.vt, self.uty = s, vt, u.T @ (y - self.y_mean)

    def coef(self, alpha):
        return self.vt.T @ (self.s / (self.s ** 2 + alpha) * self.uty)

    def predict(self, x, alpha):
        return (x - self.mu) / self.sd @ self.coef(alpha) + self.y_mean


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def metrics(y, p):
    err = p - y
    return {'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
            'spearman': spearman(y, p), 'n': int(y.size)}


def fit_subset(train, val, test, subset):
    def pick(d):
        return d['glaucoma'] if subset == 'glaucoma' else np.ones(d['md'].size, dtype=bool)

    tr, va = pick(train), pick(val)
    ridge = RidgePath(train['x'][tr], train['md'][tr])
    grid = []
    for alpha in ALPHAS:
        grid.append(dict(metrics(val['md'][va], ridge.predict(val['x'][va], alpha)),
                         alpha=alpha))
    best = min(grid, key=lambda g: (g['rmse'], -g['alpha']))
    train_mean = float(train['md'][tr].mean())
    class_mean = {True: float(train['md'][train['glaucoma']].mean()),
                  False: float(train['md'][~train['glaucoma']].mean())}

    def evaluate(d, mask):
        y = d['md'][mask]
        pred = ridge.predict(d['x'][mask], best['alpha'])
        cls = np.where(d['glaucoma'][mask], class_mean[True], class_mean[False])
        return ({'ridge': metrics(y, pred),
                 'baseline_train_mean': metrics(y, np.full(y.size, train_mean)),
                 'diagnostic_class_mean': metrics(y, cls)},
                {'y': y, 'pred': pred, 'names': [n for n, m in zip(d['names'], mask) if m],
                 'strata': d['glaucoma'][mask]})

    out = {'subset': subset, 'n_train': int(tr.sum()), 'alpha_grid': grid,
           'alpha': best['alpha'], 'alpha_rule': 'min validation RMSE; ties -> larger alpha',
           'train_mean_md': train_mean, 'class_mean_md': {'glaucoma': class_mean[True],
                                                          'non_glaucoma': class_mean[False]}}
    out['validation'], cases = evaluate(val, va)
    out['validation_note'] = 'alpha selected on this split: optimistic'
    if test is not None:
        out['test'], cases = evaluate(test, pick(test))
    return out, cases


def paired_bootstrap(a, b, n_boot, seed, stratified, chunk=500):
    """CI of metric(a) - metric(b) on identical resampled cases."""
    if a['names'] != b['names'] or not np.array_equal(a['y'], b['y']):
        raise cr_stats.IntegrityError("paired contrast on different cases")
    y, n = a['y'], a['y'].size
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(a['strata'] == v) for v in (True, False)] if stratified \
        else [np.arange(n)]
    diffs = {k: [] for k in ('mae', 'rmse', 'spearman')}
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = np.concatenate([g[rng.integers(0, g.size, size=(size, g.size))] for g in groups
                              if g.size], axis=1)
        yy, pa, pb = y[idx], a['pred'][idx], b['pred'][idx]
        ea, eb = pa - yy, pb - yy
        diffs['mae'].append(np.abs(ea).mean(1) - np.abs(eb).mean(1))
        diffs['rmse'].append(np.sqrt((ea ** 2).mean(1)) - np.sqrt((eb ** 2).mean(1)))
        ry = rankdata(yy, axis=1)
        ra, rb = rankdata(pa, axis=1), rankdata(pb, axis=1)

        def rowcorr(u, v):
            u = u - u.mean(1, keepdims=True)
            v = v - v.mean(1, keepdims=True)
            return (u * v).sum(1) / np.sqrt((u ** 2).sum(1) * (v ** 2).sum(1))

        diffs['spearman'].append(rowcorr(ry, ra) - rowcorr(ry, rb))
    point_a, point_b = metrics(y, a['pred']), metrics(y, b['pred'])
    out = {}
    for k, v in diffs.items():
        v = np.concatenate(v)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan)
        pa_, pb_ = point_a[k], point_b[k]
        out[k] = {'a': pa_, 'b': pb_,
                  'delta': None if pa_ is None or pb_ is None else pa_ - pb_,
                  'ci95': [float(lo), float(hi)]}
    return out


def _check_runs(run_dirs, split, allow_before_freeze, epoch=cr_stats.ENDPOINT_EPOCH):
    """Identity, seal binding and eligibility, all before any feature or MD value is read."""
    before = cr_stats.check_freeze(allow_before_freeze) if split == 'Test' else None
    rows = cr_stats.inventory_rows(run_dirs)
    errors = cr_stats.check_identities(rows)
    if errors:
        raise cr_stats.IntegrityError("inventory refused:\n  " + "\n  ".join(errors))
    runs, excluded, slots = {}, {}, {}
    for run_dir in sorted(set(r['run_dir'] for r in rows)):
        res = cr_stats.load_run(run_dir)
        ident = cr_stats._identity(res)
        if ident.get('epoch') != epoch:
            raise cr_stats.IntegrityError("%s: epoch %r is not %d" % (run_dir, ident.get('epoch'),
                                                                    epoch))
        if ident.get('role') == 'new' and ident.get('train_seed') not in cr_stats.REGISTERED_SEEDS:
            raise cr_stats.IntegrityError("%s: train seed %r is not registered"
                                          % (run_dir, ident.get('train_seed')))
        slot = (ident.get('arm'), ident.get('role'), ident.get('train_seed'))
        if slot in slots:
            raise cr_stats.IntegrityError("two runs for %s: %s and %s" % (slot, slots[slot], run_dir))
        slots[slot] = run_dir
        # Manifest JSON only (no prediction array): binds results.json to the seal.
        manifest, manifest_sha = cr_stats.read_seal_manifest(run_dir)
        if split == 'Test':
            cr_stats.check_receipt(run_dir, manifest_sha)
        cr_stats.check_seal_identity(run_dir, ident, manifest)
        if split == 'Test':
            _, _, preds = cr_stats.load_sealed(run_dir)
            cr_stats.check_seal_coverage(run_dir, res, preds)
        states = [((res.get('cache_provenance') or {}).get(s) or {}).get('primary', {}).get('state')
                  for s in ('Training', 'Validation', split)]
        if any(state not in ('cold', 'warm') for state in states):
            excluded[os.path.basename(run_dir)] = (
                'cache without verified per-case identity (%s); no MD regression' % states)
            continue
        runs[run_dir] = {'results': res, 'identity': ident,
                         'eligibility': cr_stats.completion_eligibility(run_dir, res, manifest)}
    return runs, excluded, before


def cmd_md(run_dirs, split='Validation', metadata=METADATA, n_boot=N_BOOT, boot_seed=BOOT_SEED,
           out=None, allow_before_freeze=False):
    split = {'validation': 'Validation', 'test': 'Test'}.get(split.lower(), split)
    if split not in ('Validation', 'Test'):
        raise cr_stats.IntegrityError("split must be validation or test")
    runs, excluded, before = _check_runs(run_dirs, split, allow_before_freeze)
    splits = ('Training', 'Validation') + (('Test',) if split == 'Test' else ())
    allowed = set(SPLIT_USE[s] for s in splits)
    per_run, cases = {}, {}
    for run_dir, run in runs.items():
        res = run['results']
        loaded = {s: load_split(res, s, run['identity']) for s in splits}
        meta = load_metadata(metadata, [n for s in splits for n in loaded[s][2]], allowed)
        data = {s: _attach_md(loaded[s][0], loaded[s][1], loaded[s][2], meta, s, run_dir)
                for s in splits}
        entry = {'identity': run['identity'], 'eligibility': run['eligibility'], 'subsets': {}}
        for subset in ('glaucoma', 'all'):
            entry['subsets'][subset], cases[(run_dir, subset)] = fit_subset(
                data['Training'], data['Validation'], data.get('Test'), subset)
        per_run[os.path.basename(run_dir)] = entry
    # Within-seed contrasts use only runs that finished before the freeze (amendment 1).
    new = {}
    for run_dir, run in runs.items():
        ident = run['identity']
        if ident.get('role') == 'new' and run['eligibility']['eligible']:
            new[(ident['arm'], int(ident['train_seed']))] = run_dir
    contrasts = []
    for seed in sorted(set(s for _, s in new)):
        r = new.get(('random', seed))
        for arm in CONTRAST_ARMS:
            a = new.get((arm, seed))
            if r is None or a is None:
                continue
            for subset in ('glaucoma', 'all'):
                contrasts.append({
                    'kind': '%s-random' % arm, 'train_seed': seed, 'subset': subset,
                    'primary': subset == 'glaucoma' and arm != 'random_cb',
                    'a': os.path.basename(a), 'b': os.path.basename(r),
                    'metrics': paired_bootstrap(cases[(a, subset)], cases[(r, subset)], n_boot,
                                                boot_seed, stratified=(subset == 'all'))})
    payload = {
        'generated': cr_stats._now().isoformat(timespec='seconds'), 'split': split,
        'before_freeze': before, 'metadata': metadata, 'n_boot': n_boot, 'boot_seed': boot_seed,
        'features': 'patch-token mean per B-scan, mean over B-scans (primary pooling)',
        'primary': 'glaucoma subset; contrasts within seed index vs RANDOM',
        'note': 'severity endpoint on the same cohort, not an independent task; '
                'lower MAE/RMSE and higher Spearman are better',
        'runs': per_run, 'excluded_runs': excluded, 'contrasts': contrasts}
    if out:
        cr_stats._write_json(out, payload)
    shown = 'test' if split == 'Test' else 'validation'
    print('%-28s %-9s %-8s %-7s %-7s %-8s %-7s %-7s' % ('run', 'subset', 'alpha', 'MAE',
                                                        'RMSE', 'Spearman', 'MAE_tm', 'MAE_cm'))
    for name, entry in per_run.items():
        for subset, block in entry['subsets'].items():
            m = block[shown]
            sp = m['ridge']['spearman']
            print('%-28s %-9s %-8g %-7.3f %-7.3f %-8s %-7.3f %-7.3f' % (
                name[:28], subset, block['alpha'], m['ridge']['mae'], m['ridge']['rmse'],
                '%.3f' % sp if sp is not None else '-', m['baseline_train_mean']['mae'],
                m['diagnostic_class_mean']['mae']))
    for c in contrasts:
        m = c['metrics']
        print('  %-18s s%-5d %-9s dMAE %+.3f [%+.3f,%+.3f]  dSpearman %s' % (
            c['kind'], c['train_seed'], c['subset'], m['mae']['delta'], m['mae']['ci95'][0],
            m['mae']['ci95'][1], '%+.3f' % m['spearman']['delta']
            if m['spearman']['delta'] is not None else '-'))
    for name, reason in excluded.items():
        print('  excluded %s: %s' % (name, reason))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--runs', nargs='+', required=True, help='probe dirs or globs')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--metadata', default=METADATA)
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--boot-seed', type=int, default=BOOT_SEED)
    parser.add_argument('--out')
    parser.add_argument('--allow-before-freeze', action='store_true', help='tests only')
    args = parser.parse_args(argv)
    try:
        cmd_md(cr_stats._expand(args.runs), args.split, args.metadata, args.n_boot,
               args.boot_seed, args.out, args.allow_before_freeze)
    except cr_stats.IntegrityError as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
