"""Seed-aware statistics for the camera-ready replication campaign (cr_seed_v1).

Implements the analysis pre-registered in
paper/genai4health2026/PREREGISTRATION_cr_seed_v1.md (sections 3-6) over probe
directories written by ``src/eval_downstream.py`` (schema ``cr_probe_v1``).

Subcommands
  inventory  table keyed by run UUID / checkpoint SHA-256 / arm / train seed /
             epoch / pooling variant / head seed; refuses duplicate or conflicting
             identities instead of silently overwriting one run with another.
  gate       G2: validation-only comparison of a new run with its anchor
             (delta of head-seed-mean validation AUC; flag |delta| > 0.01).
  unseal     verify the SHA-256 of every sealed test-prediction file, then
             compute test metrics. Use once, after the results freeze.
  final      primary analysis after the freeze: per-seed paired, label-stratified
             test-case bootstrap CIs for CENTROID-RANDOM and ENVELOPE-RANDOM
             within each seed, RANDOM-CB contrasts, outcome labels; no
             seed-level bootstrap. Seeds 1234, 5678 and (amendment 1) 9012; a seed
             index is primary only if RANDOM, CENTROID and ENVELOPE all completed.

Every subcommand writes JSON and prints a table. ``gate`` and ``inventory``
never open test predictions. ``unseal`` and ``final`` refuse to run before the
freeze (2026-10-22 08:00 PDT) unless ``--allow-before-freeze`` (tests only).

Examples
  python autopilot/cr_stats.py inventory --runs D:/jepa_phase0/runs/cr_seed_v1_probe_* --out inv.json
  python autopilot/cr_stats.py gate --run <new probe dir> --anchor <anchor probe dir>
  python autopilot/cr_stats.py unseal --run <probe dir>
  python autopilot/cr_stats.py final --inventory inv.json --out final.json
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

SCHEMA = 'cr_probe_v1'
FREEZE = '2026-10-22T08:00:00-07:00'
PRIMARY_VARIANT = 'patchmean_slicemean'
ARMS = ('random', 'centroid', 'envelope', 'random_cb')
GUIDED = ('centroid', 'envelope')
ARM_ALIASES = {'oracle': 'centroid', 'mirage': 'envelope', 'random-cb': 'random_cb',
               'randomcb': 'random_cb'}
N_BOOT = 10000
BOOT_SEED = 20261022
GATE_THRESHOLD = 0.01
# Pre-registered design (PREREGISTRATION_cr_seed_v1.md sections 1-3, amendment 1 = seed 9012).
REGISTERED_SEEDS = (1234, 5678, 9012)
PRIMARY_ARMS = ('random', 'centroid', 'envelope')
REGISTERED_HEAD_SEEDS = (42, 43, 44, 45, 46)
ENDPOINT_EPOCH = 50
IDENTITY_KEYS = ('run_uuid', 'checkpoint_sha256', 'arm', 'train_seed', 'role', 'epoch')


class IntegrityError(RuntimeError):
    """Identity, hash or protocol violation: the analysis must not proceed."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _now():
    return datetime.datetime.now().astimezone()


def norm_arm(arm):
    if arm is None:
        return None
    arm = ARM_ALIASES.get(str(arm).lower(), str(arm).lower())
    if arm not in ARMS:
        raise IntegrityError("Unknown arm %r" % arm)
    return arm


def check_freeze(allow_before_freeze=False, now=None):
    now = now or _now()
    freeze = datetime.datetime.fromisoformat(FREEZE)
    before = now < freeze
    if before and not allow_before_freeze:
        raise IntegrityError("Refusing to open sealed test predictions before the results "
                             "freeze (%s)" % FREEZE)
    return before


def load_run(run_dir):
    path = os.path.join(run_dir, 'results.json')
    with open(path, 'r', encoding='utf-8') as stream:
        res = json.load(stream)
    if res.get('schema') != SCHEMA or res.get('status') != 'complete':
        raise IntegrityError("%s is not a completed %s probe" % (run_dir, SCHEMA))
    res['_run_dir'] = os.path.realpath(run_dir)
    return res


def _identity(res):
    ident = dict(res.get('identity') or {})
    ident['arm'] = norm_arm(ident.get('arm'))
    return ident


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def inventory_rows(run_dirs):
    rows = []
    for run_dir in run_dirs:
        res = load_run(run_dir)
        ident = _identity(res)
        run_uuid = ident.get('run_uuid')
        derived = False
        if not run_uuid and ident.get('role') == 'anchor' and ident.get('checkpoint_sha256'):
            # Original anchors have no training-run UUID; key them by arm/epoch/checkpoint.
            run_uuid = 'anchor-%s-ep%s-%s' % (ident.get('arm'), ident.get('epoch'),
                                              ident['checkpoint_sha256'][:12])
            derived = True
        for variant, summary in sorted(res['variants'].items()):
            for rec in summary['per_seed']:
                rows.append({
                    'run_dir': res['_run_dir'], 'run_uuid': run_uuid,
                    'run_uuid_derived': derived,
                    'checkpoint_sha256': ident.get('checkpoint_sha256'),
                    'arm': ident.get('arm'), 'train_seed': ident.get('train_seed'),
                    'role': ident.get('role'), 'epoch': ident.get('epoch'),
                    'variant': variant, 'head_seed': rec['head_seed'],
                    'val_auc': rec['best_val_auc'], 'best_epoch': rec['best_epoch'],
                    'sealed': bool(res.get('sealed')),
                    'test_predictions': rec.get('test_predictions'),
                    'test_predictions_sha256': rec.get('test_predictions_sha256')})
    return rows


def check_identities(rows):
    """Return a list of identity violations (empty when the inventory is consistent)."""
    errors = []
    for row in rows:
        missing = [k for k in ('run_uuid', 'checkpoint_sha256', 'arm', 'role', 'epoch')
                   if row.get(k) in (None, '')]
        if row.get('role') == 'new' and row.get('train_seed') is None:
            missing.append('train_seed')
        if missing:
            errors.append("%s: missing identity fields %s" % (row['run_dir'], sorted(set(missing))))
    seen = {}
    for row in rows:
        key = (row['run_uuid'], row['checkpoint_sha256'], row['arm'], row['train_seed'],
               row['epoch'], row['variant'], row['head_seed'])
        if key in seen:
            errors.append("duplicate identity %s in %s and %s" % (key, seen[key], row['run_dir']))
        else:
            seen[key] = row['run_dir']
    by_ckpt, by_slot, by_uuid = {}, {}, {}
    for row in rows:
        label = (row['arm'], row['train_seed'], row['run_uuid'], row['epoch'])
        by_ckpt.setdefault(row['checkpoint_sha256'], set()).add(label)
        slot = (row['arm'], row['train_seed'], row['role'], row['epoch'])
        by_slot.setdefault(slot, set()).add(row['checkpoint_sha256'])
        by_uuid.setdefault(row['run_uuid'], set()).add(row['checkpoint_sha256'])
    for ckpt, labels in by_ckpt.items():
        if len(labels) > 1:
            errors.append("checkpoint %s carries conflicting labels %s" % (ckpt, sorted(
                labels, key=str)))
    for slot, ckpts in by_slot.items():
        if len(ckpts) > 1:
            errors.append("arm/seed/role/epoch %s maps to several checkpoints %s"
                          % (slot, sorted(ckpts, key=str)))
    for uuid, ckpts in by_uuid.items():
        if len(ckpts) > 1:
            errors.append("run %s maps to several checkpoints %s" % (uuid, sorted(ckpts, key=str)))
    return errors


def run_summaries(rows):
    out = {}
    for row in rows:
        key = (row['run_dir'], row['variant'])
        rec = out.setdefault(key, dict((k, row[k]) for k in (
            'run_dir', 'run_uuid', 'arm', 'train_seed', 'role', 'epoch', 'checkpoint_sha256',
            'variant', 'sealed')))
        rec.setdefault('val_aucs', []).append(row['val_auc'])
        rec.setdefault('head_seeds', []).append(row['head_seed'])
    table = []
    for rec in out.values():
        vals = np.asarray(rec['val_aucs'], dtype=np.float64)
        rec['val_auc_mean'] = float(vals.mean())
        rec['val_auc_sd'] = float(vals.std(ddof=1)) if vals.size > 1 else None
        table.append(rec)
    return sorted(table, key=lambda r: (str(r['arm']), str(r['role']), str(r['train_seed']),
                                        r['variant']))


def cmd_inventory(run_dirs, out=None):
    rows = inventory_rows(run_dirs)
    errors = check_identities(rows)
    if errors:
        raise IntegrityError("inventory refused:\n  " + "\n  ".join(errors))
    result = {'generated': _now().isoformat(timespec='seconds'), 'n_runs': len(set(
        r['run_dir'] for r in rows)), 'rows': rows, 'runs': run_summaries(rows)}
    if out:
        _write_json(out, result)
    print('%-9s %-6s %-6s %-5s %-24s %-3s %-8s %-8s %-6s %-12s %s' % (
        'arm', 'seed', 'role', 'ep', 'variant', 'n', 'val_mean', 'val_sd', 'sealed', 'ckpt',
        'run_uuid'))
    for r in result['runs']:
        print('%-9s %-6s %-6s %-5s %-24s %-3d %.4f   %-8s %-6s %-12s %s' % (
            r['arm'], r['train_seed'], r['role'], r['epoch'], r['variant'], len(r['val_aucs']),
            r['val_auc_mean'], '%.4f' % r['val_auc_sd'] if r['val_auc_sd'] is not None else '-',
            r['sealed'], str(r['checkpoint_sha256'])[:12], r['run_uuid']))
    return result


# ---------------------------------------------------------------------------
# gate (validation only)
# ---------------------------------------------------------------------------

def _val_aucs(res, variant):
    """Per-head-seed validation AUC, recomputed from the saved predictions."""
    out, labels = {}, None
    for rec in res['variants'][variant]['per_seed']:
        path = os.path.join(res['_run_dir'], rec['val_predictions'])
        with np.load(path) as z:
            y, p = z['labels'].astype(np.int64), z['probs'].astype(np.float64)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise IntegrityError("validation labels differ across head seeds in %s" % path)
        auc = float(roc_auc_score(y, p))
        if abs(auc - rec['best_val_auc']) > 1e-4:
            raise IntegrityError("%s: recomputed val AUC %.6f != recorded %.6f"
                                 % (path, auc, rec['best_val_auc']))
        out[rec['head_seed']] = auc
    return out, labels


def gate_pair(run_dir, anchor_dir, threshold=GATE_THRESHOLD, allow_arm_mismatch=False,
              endpoint_epoch=ENDPOINT_EPOCH):
    run, anchor = load_run(run_dir), load_run(anchor_dir)
    ri, ai = _identity(run), _identity(anchor)
    if ri.get('arm') != ai.get('arm') and not allow_arm_mismatch:
        raise IntegrityError("gate compares the same arm only: run %s vs anchor %s"
                             % (ri.get('arm'), ai.get('arm')))
    if endpoint_epoch is not None and (ri.get('epoch') != endpoint_epoch
                                       or ai.get('epoch') != endpoint_epoch):
        raise IntegrityError("gate is defined at epoch %d: run epoch %r, anchor epoch %r"
                             % (endpoint_epoch, ri.get('epoch'), ai.get('epoch')))
    if ri.get('checkpoint_sha256') == ai.get('checkpoint_sha256'):
        raise IntegrityError("run and anchor share checkpoint %s" % ri.get('checkpoint_sha256'))
    variants = {}
    for variant in [v for v in run['variants'] if v in anchor['variants']]:
        rv, ry = _val_aucs(run, variant)
        av, ay = _val_aucs(anchor, variant)
        if not np.array_equal(ry, ay):
            raise IntegrityError("validation labels differ between run and anchor")
        if sorted(rv) != sorted(av):
            raise IntegrityError("head seeds differ: %s vs %s" % (sorted(rv), sorted(av)))
        r_mean, a_mean = float(np.mean(list(rv.values()))), float(np.mean(list(av.values())))
        delta = r_mean - a_mean
        variants[variant] = {
            'run_val_auc_mean': r_mean, 'anchor_val_auc_mean': a_mean, 'delta': delta,
            'run_val_auc_sd': float(np.std(list(rv.values()), ddof=1)) if len(rv) > 1 else None,
            'anchor_val_auc_sd': float(np.std(list(av.values()), ddof=1)) if len(av) > 1 else None,
            'run_per_seed': rv, 'anchor_per_seed': av,
            'flag': bool(abs(delta) > threshold)}
    if PRIMARY_VARIANT not in variants:
        raise IntegrityError("primary variant missing from run or anchor")
    return {'run': run['_run_dir'], 'anchor': anchor['_run_dir'], 'arm': ri.get('arm'),
            'run_identity': ri, 'anchor_identity': ai, 'threshold': threshold,
            'validation_only': True, 'primary_variant': PRIMARY_VARIANT,
            'primary_delta': variants[PRIMARY_VARIANT]['delta'],
            'primary_flag': variants[PRIMARY_VARIANT]['flag'], 'variants': variants,
            'decision_rule': 'G2: |delta| > threshold triggers investigation (loss curves, mask '
                             'statistics, other arms), never a rerun because of AUC'}


def _gate_pairs_from_inventory(inv):
    runs = {}
    for row in inv['rows']:
        runs[row['run_dir']] = row
    anchors = {}
    for run_dir, row in runs.items():
        if row['role'] == 'anchor':
            if row['arm'] in anchors:
                raise IntegrityError("several anchors for arm %s" % row['arm'])
            anchors[row['arm']] = run_dir
    pairs, unpaired = [], []
    for run_dir, row in sorted(runs.items()):
        if row['role'] != 'new':
            continue
        if row['arm'] in anchors:
            pairs.append((run_dir, anchors[row['arm']]))
        else:
            unpaired.append(run_dir)
    return pairs, unpaired


def cmd_gate(pairs, threshold=GATE_THRESHOLD, out=None, allow_arm_mismatch=False,
             unpaired=(), endpoint_epoch=ENDPOINT_EPOCH):
    results = [gate_pair(r, a, threshold, allow_arm_mismatch, endpoint_epoch) for r, a in pairs]
    payload = {'generated': _now().isoformat(timespec='seconds'), 'threshold': threshold,
               'validation_only': True, 'gates': results, 'unpaired_runs': list(unpaired)}
    if out:
        _write_json(out, payload)
    print('%-9s %-24s %-9s %-9s %-9s %s' % ('arm', 'variant', 'run', 'anchor', 'delta', 'flag'))
    for g in results:
        for variant, v in g['variants'].items():
            print('%-9s %-24s %.4f    %.4f    %+.4f   %s  (%s)' % (
                g['arm'], variant, v['run_val_auc_mean'], v['anchor_val_auc_mean'], v['delta'],
                'FLAG' if v['flag'] else 'ok', os.path.basename(g['run'])))
    for run_dir in unpaired:
        print('  no anchor for %s' % run_dir)
    return payload


# ---------------------------------------------------------------------------
# unseal
# ---------------------------------------------------------------------------

def read_seal_manifest(run_dir):
    """Hash-verified sealed manifest JSON only; no prediction array is opened."""
    manifest_path = os.path.join(run_dir, 'sealed_manifest.json')
    sidecar = manifest_path + '.sha256'
    if not os.path.exists(manifest_path) or not os.path.exists(sidecar):
        raise IntegrityError("%s has no sealed manifest (+ .sha256)" % run_dir)
    with open(sidecar, 'r', encoding='utf-8') as stream:
        recorded = stream.read().split()[0]
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != recorded:
        raise IntegrityError("sealed_manifest.json hash mismatch in %s" % run_dir)
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        return json.load(stream), manifest_sha


def check_receipt(run_dir, manifest_sha):
    """The unseal receipt must exist and belong to the current seal (checked before arrays)."""
    receipt_path = os.path.join(run_dir, 'unsealed_results.json')
    if not os.path.exists(receipt_path):
        raise IntegrityError("%s has not been unsealed (run `unseal` first)" % run_dir)
    with open(receipt_path, 'r', encoding='utf-8') as stream:
        if json.load(stream).get('sealed_manifest_sha256') != manifest_sha:
            raise IntegrityError("%s: unseal receipt is not bound to the current seal" % run_dir)


def check_seal_identity(run_dir, ident, manifest):
    """The (unhashed) results.json identity must equal the identity under the seal."""
    sealed_ident = dict(manifest.get('identity') or {})
    sealed_ident['arm'] = norm_arm(sealed_ident.get('arm'))
    diff = [k for k in IDENTITY_KEYS if sealed_ident.get(k) != ident.get(k)]
    if diff:
        raise IntegrityError("%s: results.json identity contradicts the seal on %s"
                             % (run_dir, diff))
    return sealed_ident


def check_seal_coverage(run_dir, res, preds):
    """results.json per-seed predictions must be exactly the sealed files."""
    listed = set((v, int(r['head_seed']), os.path.normpath(r['test_predictions']),
                  r['test_predictions_sha256'])
                 for v, s in res['variants'].items() for r in s['per_seed'])
    sealed = set((v, s, os.path.normpath(p['rel_path']), p['sha256'])
                 for (v, s), p in preds.items())
    if listed != sealed:
        raise IntegrityError("%s: results.json predictions differ from the sealed files" % run_dir)


def _parse_time(text):
    try:
        stamp = datetime.datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    except ValueError:
        try:
            stamp = datetime.datetime.strptime(str(text), '%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=datetime.timezone.utc)


def completion_eligibility(run_dir, res, manifest, epoch=ENDPOINT_EPOCH):
    """Amendment 1: a run counts only if it FINISHED before the freeze.

    Evidence: the probe's seal creation time (hash-bound) and results.json finish time;
    for new runs also the trainer's epoch-50 stop file, which must be the hash-recorded
    run provenance of the probe, list the probed checkpoint and predate the freeze.
    Missing or late evidence makes the run ineligible (reported descriptively).
    """
    freeze = datetime.datetime.fromisoformat(FREEZE)
    ident = _identity(res)
    times = {'seal_created': manifest.get('created'), 'probe_finished': res.get('finished')}
    reasons = []
    if ident.get('role') == 'new':
        path = ident.get('run_provenance')
        stop = None
        if not path or not os.path.exists(path):
            reasons.append('no epoch-%d stop file' % epoch)
        elif sha256_file(path) != ident.get('run_provenance_sha256'):
            reasons.append('stop file changed since the probe')
        else:
            with open(path, 'r', encoding='utf-8') as stream:
                stop = json.load(stream)
            listed = [c.get('sha256') for c in stop.get('checkpoints') or []]
            if stop.get('schema') != 'jepa_stop_epoch_v1' or stop.get('epoch') != epoch \
                    or ident.get('checkpoint_sha256') not in listed:
                reasons.append('stop file does not certify the probed epoch-%d checkpoint'
                               % epoch)
            times['stop_file'] = stop.get('timestamp_utc')
    for name, text in times.items():
        stamp = _parse_time(text)
        if stamp is None:
            reasons.append('missing %s time' % name)
        elif stamp >= freeze:
            reasons.append('%s after the freeze (%s)' % (name, text))
    return {'eligible': not reasons, 'reasons': reasons, 'times': times}


def load_sealed(run_dir):
    """Verify the sealed manifest and every file hash; return predictions by (variant, seed)."""
    manifest_path = os.path.join(run_dir, 'sealed_manifest.json')
    manifest, manifest_sha = read_seal_manifest(run_dir)
    label_id = manifest['test_label_identity']
    preds = {}
    for entry in manifest['files']:
        key = (entry['variant'], int(entry['head_seed']))
        if key in preds:
            raise IntegrityError("duplicate sealed entry %s in %s" % (key, manifest_path))
        path = os.path.join(run_dir, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise IntegrityError("sealed file hash mismatch: %s" % path)
        side = os.path.join(run_dir, entry['sidecar_path'])
        if sha256_file(side) != entry['sidecar_sha256']:
            raise IntegrityError("sealed sidecar hash mismatch: %s" % side)
        head = os.path.join(os.path.dirname(path), 'best_model.pt')
        if os.path.exists(head) and sha256_file(head) != entry.get('head_checkpoint_sha256'):
            raise IntegrityError("head checkpoint hash mismatch: %s" % head)
        with np.load(path) as z:
            labels = z['labels'].astype(np.int8)
            probs = z['probs']
            subjects = [str(s) for s in z['subject_ids']]
        if hashlib.sha256(labels.tobytes()).hexdigest() != label_id['labels_int8_sha256']:
            raise IntegrityError("test labels in %s differ from the sealed label identity" % path)
        if _sha256_text('\n'.join(subjects)) != label_id['subject_ids_sha256']:
            raise IntegrityError("test case order in %s differs from the sealed identity" % path)
        preds[key] = {
            'labels': labels.astype(np.int64), 'probs': probs, 'path': path,
            'sha256': entry['sha256'], 'rel_path': entry['path']}
    return manifest, manifest_sha, preds


def _binary_metrics(labels, probs):
    pred = (probs >= 0.5).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {'test_auc': float(roc_auc_score(labels, probs)),
            'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1)}


def cmd_unseal(run_dir, allow_before_freeze=False):
    before = check_freeze(allow_before_freeze)
    manifest, manifest_sha, preds = load_sealed(run_dir)
    receipt_path = os.path.join(run_dir, 'unsealed_results.json')
    if os.path.exists(receipt_path):
        # Opened once: a repeat call re-verifies the seal and returns the first receipt.
        with open(receipt_path, 'r', encoding='utf-8') as stream:
            receipt = json.load(stream)
        if receipt.get('sealed_manifest_sha256') != manifest_sha:
            raise IntegrityError("%s does not belong to the current sealed manifest" % receipt_path)
        with open(os.path.join(run_dir, 'unseal_log.jsonl'), 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'reverified_at': _now().isoformat(timespec='seconds'),
                                     'sealed_manifest_sha256': manifest_sha}) + '\n')
        print('already unsealed at %s; seal re-verified, receipt unchanged'
              % receipt.get('unsealed_at'))
        return receipt
    variants = {}
    for (variant, seed), rec in sorted(preds.items()):
        metrics = _binary_metrics(rec['labels'], rec['probs'])
        variants.setdefault(variant, {'per_seed': []})['per_seed'].append(
            dict(metrics, head_seed=seed))
    for summary in variants.values():
        aucs = np.array([r['test_auc'] for r in summary['per_seed']])
        summary['test_auc_mean'] = float(aucs.mean())
        summary['test_auc_sd'] = float(aucs.std(ddof=1)) if aucs.size > 1 else None
    payload = {'run_dir': os.path.realpath(run_dir), 'unsealed_at': _now().isoformat(
        timespec='seconds'), 'unsealed_before_freeze': before, 'freeze': FREEZE,
        'sealed_manifest_sha256': manifest_sha, 'identity': manifest.get('identity'),
        'variants': variants}
    _write_json(os.path.join(run_dir, 'unsealed_results.json'), payload)
    with open(os.path.join(run_dir, 'unseal_log.jsonl'), 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'unsealed_at': payload['unsealed_at'],
                                 'before_freeze': before,
                                 'sealed_manifest_sha256': manifest_sha}) + '\n')
    print('%-24s %-5s %-8s %-8s %s' % ('variant', 'seed', 'AUC', 'sens', 'spec'))
    for variant, summary in variants.items():
        for r in summary['per_seed']:
            print('%-24s %-5d %.4f   %.4f   %.4f' % (variant, r['head_seed'], r['test_auc'],
                                                   r['sensitivity'], r['specificity']))
        print('%-24s mean  %.4f   sd %s' % (variant, summary['test_auc_mean'],
                                            '%.4f' % summary['test_auc_sd']
                                            if summary['test_auc_sd'] is not None else '-'))
    return payload


# ---------------------------------------------------------------------------
# final (after the freeze)
# ---------------------------------------------------------------------------

def bootstrap_weights(labels, n_boot, rng):
    """Label-stratified case-resampling counts, (n_boot, n); shared by every model."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    weights = np.zeros((n_boot, labels.size), dtype=np.float64)
    weights[:, pos] = rng.multinomial(pos.size, np.full(pos.size, 1.0 / pos.size), size=n_boot)
    weights[:, neg] = rng.multinomial(neg.size, np.full(neg.size, 1.0 / neg.size), size=n_boot)
    return weights


class WeightedAUC(object):
    """Mann-Whitney AUC under case weights (ties count 1/2) for many weight rows."""

    def __init__(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64)
        self.order = np.argsort(scores, kind='mergesort')
        ordered = scores[self.order]
        self.starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        y = np.asarray(labels)[self.order]
        self.pos, self.neg = (y == 1).astype(np.float64), (y == 0).astype(np.float64)

    def __call__(self, weights):
        w = weights[:, self.order]
        gp = np.add.reduceat(w * self.pos, self.starts, axis=1)
        gn = np.add.reduceat(w * self.neg, self.starts, axis=1)
        below = np.cumsum(gn, axis=1) - gn
        return (gp * (below + 0.5 * gn)).sum(axis=1) / (gp.sum(axis=1) * gn.sum(axis=1))


def outcome_label(deltas, random_values):
    """PREREGISTRATION section 3 / amendment 1 outcome label for one guided arm.

    ``deltas`` are the guided-minus-RANDOM deltas of the COMPLETE seed indices only (all of
    RANDOM, CENTROID and ENVELOPE finished); ``random_values`` are RANDOM's realizations
    used for the range: the original plus the RANDOM runs of those complete indices.
    Directional labels need at least two complete indices and the original RANDOM; with
    two indices this is section 3 unchanged, with three amendment 1 (all three deltas > 0
    and mean delta > range of the four realizations). Anything less is 'incomplete'.
    """
    if not deltas:
        return 'not_available'
    if len(deltas) < 2 or len(random_values) != 1 + len(deltas):
        return 'incomplete'
    mean = float(np.mean(deltas))
    if mean <= 0:
        return 'reversed'
    spread = max(random_values) - min(random_values)
    if all(d > 0 for d in deltas) and mean > spread:
        return 'consistent_direction'
    return 'within_run_to_run_variation'


def _recipe(res):
    """Probe recipe that must be identical across every run in the registered analysis."""
    cfg = res.get('config')
    if not isinstance(cfg, dict):
        raise IntegrityError("%s: results.json has no rendered config" % res['_run_dir'])
    data = {k: v for k, v in (cfg.get('data') or {}).items() if k != 'legacy_cache_policy'}
    model = {k: v for k, v in (cfg.get('model') or {}).items() if k != 'encoder_checkpoint'}
    return {'mode': cfg.get('mode'), 'data': data, 'model': model,
            'training': cfg.get('training'),
            'head_seeds': sorted((cfg.get('probe') or {}).get('head_seeds') or [])}


def _collect_final_inputs(run_dirs, allow_before_freeze, head_seeds=REGISTERED_HEAD_SEEDS,
                          seeds=REGISTERED_SEEDS, epoch=ENDPOINT_EPOCH):
    rows = inventory_rows(run_dirs)
    errors = check_identities(rows)
    if errors:
        raise IntegrityError("inventory refused:\n  " + "\n  ".join(errors))
    runs, labels, subjects, recipe, slots = {}, None, None, None, {}
    for run_dir in sorted(set(r['run_dir'] for r in rows)):
        res = load_run(run_dir)
        ident = _identity(res)
        # Registered endpoint and design: epoch 50 only, registered seeds, one run per slot.
        if ident.get('epoch') != epoch:
            raise IntegrityError("%s: epoch %r is not the registered endpoint %d"
                                 % (run_dir, ident.get('epoch'), epoch))
        if ident.get('role') == 'new' and ident.get('train_seed') not in seeds:
            raise IntegrityError("%s: train seed %r is not registered %s"
                                 % (run_dir, ident.get('train_seed'), seeds))
        slot = (ident.get('arm'), ident.get('role'),
                ident.get('train_seed') if ident.get('role') == 'new' else None)
        if slot in slots:
            raise IntegrityError("two runs for %s: %s and %s" % (slot, slots[slot], run_dir))
        slots[slot] = run_dir
        this_recipe = _recipe(res)
        if recipe is None:
            recipe = this_recipe
        elif this_recipe != recipe:
            raise IntegrityError("%s: probe recipe differs from the other runs" % run_dir)
        # Seal first (manifest only), then the receipt, then identity; arrays come last.
        manifest, manifest_sha = read_seal_manifest(run_dir)
        check_receipt(run_dir, manifest_sha)
        check_seal_identity(run_dir, ident, manifest)
        manifest, manifest_sha, preds = load_sealed(run_dir)
        check_seal_coverage(run_dir, res, preds)
        for variant in set(v for v, _ in preds):
            if sorted(s for v, s in preds if v == variant) != sorted(head_seeds):
                raise IntegrityError("%s: %s head seeds are not %s"
                                     % (run_dir, variant, list(head_seeds)))
        for rec in preds.values():
            if labels is None:
                labels = rec['labels']
            elif not np.array_equal(labels, rec['labels']):
                raise IntegrityError("test labels differ across runs (%s)" % run_dir)
        if subjects is None:
            subjects = manifest['test_label_identity']['subject_ids_sha256']
        elif subjects != manifest['test_label_identity']['subject_ids_sha256']:
            raise IntegrityError("test case order differs across runs (%s)" % run_dir)
        runs[run_dir] = {'identity': ident, 'results': res, 'preds': preds,
                         'eligibility': completion_eligibility(run_dir, res, manifest)}
    return runs, labels


def cmd_final(run_dirs, n_boot=N_BOOT, boot_seed=BOOT_SEED, out=None,
              allow_before_freeze=False, chunk=1000, head_seeds=REGISTERED_HEAD_SEEDS):
    before = check_freeze(allow_before_freeze)
    runs, labels = _collect_final_inputs(run_dirs, allow_before_freeze, head_seeds)
    new, anchors = {}, {}
    for run_dir, run in runs.items():
        ident = run['identity']
        if ident['role'] == 'new':
            new[(ident['arm'], int(ident['train_seed']))] = run_dir
        elif ident['role'] == 'anchor':
            if ident['arm'] in anchors:
                raise IntegrityError("several anchors for arm %s" % ident['arm'])
            anchors[ident['arm']] = run_dir

    # Secondary variants may be missing for some runs (e.g. no patch-max for an
    # anchor without weights); contrasts use only the runs that have them.
    variants = sorted(set.union(*[set(v for v, _ in run['preds']) for run in runs.values()]))
    if any(PRIMARY_VARIANT not in set(v for v, _ in run['preds']) for run in runs.values()):
        raise IntegrityError("primary variant missing from at least one run")

    point, val, boot = {}, {}, {}
    for run_dir, run in runs.items():
        for variant in variants:
            seeds = sorted(s for v, s in run['preds'] if v == variant)
            if not seeds:
                continue
            point[(run_dir, variant)] = float(np.mean(
                [roc_auc_score(labels, run['preds'][(variant, s)]['probs']) for s in seeds]))
            val[(run_dir, variant)] = run['results']['variants'][variant]['val_auc_mean']
            boot[(run_dir, variant)] = (seeds, [WeightedAUC(run['preds'][(variant, s)]['probs'],
                                                            labels) for s in seeds])
    rng = np.random.default_rng(boot_seed)
    draws = {key: np.empty(n_boot) for key in boot}
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        weights = bootstrap_weights(labels, size, rng)
        for key, (_, aucs) in boot.items():
            draws[key][start:start + size] = np.mean([auc(weights) for auc in aucs], axis=0)

    def contrast(a, b, variant, kind, seed):
        d = draws[(a, variant)] - draws[(b, variant)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {'kind': kind, 'train_seed': seed, 'variant': variant,
                'a': os.path.basename(a), 'b': os.path.basename(b),
                'auc_a': point[(a, variant)], 'auc_b': point[(b, variant)],
                'delta': point[(a, variant)] - point[(b, variant)],
                'ci95': [float(lo), float(hi)],
                'val_delta': val[(a, variant)] - val[(b, variant)]}

    seeds_new = sorted(set(s for _, s in new))
    analysis = {}
    for variant in variants:
        def has(run_dir):
            return run_dir is not None and (run_dir, variant) in point

        def ok(run_dir):
            return has(run_dir) and runs[run_dir]['eligibility']['eligible']

        # Amendment 1: a seed index enters the primary analysis only if RANDOM, CENTROID
        # and ENVELOPE all finished before the freeze; other indices are descriptive.
        complete = [s for s in seeds_new if all(ok(new.get((arm, s))) for arm in PRIMARY_ARMS)]
        primary_contrasts, matched, descriptive = [], [], []
        for seed in seeds_new:
            r = new.get(('random', seed))
            for arm in GUIDED:
                if has(r) and has(new.get((arm, seed))):
                    c = dict(contrast(new[(arm, seed)], r, variant, '%s-random' % arm, seed),
                             arm=arm)
                    if seed in complete:
                        primary_contrasts.append(c)
                    else:
                        late = not (ok(r) and ok(new[(arm, seed)]))
                        descriptive.append(dict(c, descriptive_only=True, reason=(
                            'run not finished before the freeze' if late
                            else 'seed index incomplete')))
            cb = new.get(('random_cb', seed))
            for a, b, kind in ((new.get(('centroid', seed)), cb, 'centroid-random_cb'),
                               (cb, r, 'random_cb-random')):
                if has(a) and has(b):
                    c = contrast(a, b, variant, kind, seed)
                    if ok(a) and ok(b):
                        matched.append(c)
                    else:
                        descriptive.append(dict(c, descriptive_only=True,
                                                reason='run not finished before the freeze'))
        random_runs = [anchors.get('random')] + [new.get(('random', s)) for s in complete]
        random_values = [point[(run_dir, variant)] for run_dir in random_runs if ok(run_dir)]
        spread = (max(random_values) - min(random_values)) if len(random_values) >= 2 else None
        all_random = [point[(d, variant)] for d in [anchors.get('random')] +
                      [new.get(('random', s)) for s in seeds_new] if has(d)]
        outcomes = {}
        for arm in GUIDED:
            deltas = [c['delta'] for c in primary_contrasts if c['arm'] == arm]
            sens = list(deltas)
            original = None
            if ok(anchors.get(arm)) and ok(anchors.get('random')):
                original = point[(anchors[arm], variant)] - point[(anchors['random'], variant)]
                sens.append(original)
            outcomes[arm] = {
                'new_seed_deltas': deltas, 'n_new_seeds': len(deltas),
                'complete_seed_indices': complete,
                'mean_delta': float(np.mean(deltas)) if deltas else None,
                'random_realizations': random_values, 'random_range': spread,
                'random_range_all_completed_runs': (max(all_random) - min(all_random)
                                                    if len(all_random) >= 2 else None),
                'label': outcome_label(deltas, random_values),
                'design_complete': len(deltas) >= 2 and len(random_values) == 1 + len(deltas),
                'sensitivity_with_original': {
                    'original_delta': original,
                    'pooled_mean_delta': float(np.mean(sens)) if sens else None,
                    'n': len(sens), 'descriptive_only': True}}
        for c in matched:
            c['abs_delta_exceeds_random_range'] = (abs(c['delta']) > spread
                                                   if spread is not None else None)
        analysis[variant] = {'primary_contrasts': primary_contrasts, 'outcomes': outcomes,
                             'complete_seed_indices': complete,
                             'incomplete_seed_indices': [s for s in seeds_new
                                                         if s not in complete],
                             'descriptive_incomplete_contrasts': descriptive,
                             'matched_control': matched,
                             'per_run_test_auc': {os.path.basename(k[0]): v for k, v in
                                                  point.items() if k[1] == variant},
                             'per_run_val_auc': {os.path.basename(k[0]): v for k, v in
                                                 val.items() if k[1] == variant}}
    payload = {
        'generated': _now().isoformat(timespec='seconds'), 'before_freeze': before,
        'freeze': FREEZE, 'n_test': int(labels.size), 'n_boot': n_boot, 'boot_seed': boot_seed,
        'bootstrap': 'paired, label-stratified test-case resampling; same draw for every model; '
                     'per-encoder AUC = mean over head seeds within each draw',
        'seed_level_inference': 'none (2-3 seed indices); per-seed CIs only',
        'primary_variant': PRIMARY_VARIANT, 'secondary_variants': [v for v in variants
                                                                   if v != PRIMARY_VARIANT],
        'runs': {os.path.basename(k): {'identity': v['identity'],
                                       'eligibility': v['eligibility']}
                 for k, v in runs.items()},
        'analysis': analysis}
    if out:
        _write_json(out, payload)
    for variant in [PRIMARY_VARIANT] + [v for v in variants if v != PRIMARY_VARIANT]:
        block = analysis[variant]
        print('\n[%s]%s' % (variant, ' (primary)' if variant == PRIMARY_VARIANT else ''))
        print('  %-20s %-5s %-8s %-8s %-8s %-20s %s' % ('contrast', 'seed', 'AUC_a', 'AUC_b',
                                                        'delta', '95% CI', 'val_delta'))
        for c in block['primary_contrasts'] + block['matched_control'] + \
                block['descriptive_incomplete_contrasts']:
            print('  %-20s %-5s %.4f   %.4f   %+.4f  [%+.4f, %+.4f]  %+.4f%s' % (
                c['kind'], c['train_seed'], c['auc_a'], c['auc_b'], c['delta'], c['ci95'][0],
                c['ci95'][1], c['val_delta'],
                '  (descriptive: incomplete index)' if c.get('descriptive_only') else ''))
        for arm, o in block['outcomes'].items():
            print('  outcome %-9s %-28s mean delta %s, RANDOM range %s' % (
                arm, o['label'], '%+.4f' % o['mean_delta'] if o['mean_delta'] is not None
                else '-', '%.4f' % o['random_range'] if o['random_range'] is not None else '-'))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=1, sort_keys=False, default=str)


def _expand(paths):
    out = []
    for item in paths or []:
        hits = sorted(glob.glob(item)) if any(ch in item for ch in '*?[') else [item]
        out.extend(h for h in hits if os.path.isdir(h))
    return out


def _runs_from_args(args):
    if getattr(args, 'inventory', None):
        with open(args.inventory, 'r', encoding='utf-8') as stream:
            inv = json.load(stream)
        return sorted(set(r['run_dir'] for r in inv['rows']))
    return _expand(args.runs)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory')
    p.add_argument('--runs', nargs='+', required=True, help='probe dirs or glob patterns')
    p.add_argument('--out')
    p = sub.add_parser('gate')
    p.add_argument('--run')
    p.add_argument('--anchor')
    p.add_argument('--inventory', help='pair every role=new run with its arm anchor')
    p.add_argument('--threshold', type=float, default=GATE_THRESHOLD)
    p.add_argument('--allow-arm-mismatch', action='store_true')
    p.add_argument('--endpoint-epoch', type=int, default=ENDPOINT_EPOCH,
                   help='registered endpoint; run and anchor must both be at this epoch')
    p.add_argument('--fail-on-flag', action='store_true', help='exit 3 when a gate is flagged')
    p.add_argument('--out')
    p = sub.add_parser('unseal')
    p.add_argument('--run', required=True)
    p.add_argument('--allow-before-freeze', action='store_true', help='tests only')
    p = sub.add_parser('final')
    p.add_argument('--runs', nargs='+')
    p.add_argument('--inventory')
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--boot-seed', type=int, default=BOOT_SEED)
    p.add_argument('--out')
    p.add_argument('--allow-before-freeze', action='store_true', help='tests only')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'inventory':
            cmd_inventory(_expand(args.runs), args.out)
        elif args.command == 'gate':
            if args.inventory:
                with open(args.inventory, 'r', encoding='utf-8') as stream:
                    pairs, unpaired = _gate_pairs_from_inventory(json.load(stream))
            elif args.run and args.anchor:
                pairs, unpaired = [(args.run, args.anchor)], []
            else:
                raise IntegrityError("gate needs --run and --anchor, or --inventory")
            payload = cmd_gate(pairs, args.threshold, args.out, args.allow_arm_mismatch, unpaired,
                               args.endpoint_epoch)
            if args.fail_on_flag and any(g['primary_flag'] for g in payload['gates']):
                return 3
        elif args.command == 'unseal':
            cmd_unseal(args.run, args.allow_before_freeze)
        elif args.command == 'final':
            cmd_final(_runs_from_args(args), args.n_boot, args.boot_seed, args.out,
                      args.allow_before_freeze)
    except IntegrityError as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
