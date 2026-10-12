"""autopilot/make_cr_results.py: scenario selection per the pre-registration, numbers formatted
from the inputs, refusal of unsealed or identity-incomplete inputs, no hand-typed numbers.
CPU only; synthetic fixtures from tests/cr_results_fixtures.py (real cr_stats unseal/final)."""
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'autopilot'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault('MPLBACKEND', 'Agg')
import cr_results_fixtures as fx  # noqa: E402
import cr_stats  # noqa: E402
import make_cr_results as mk  # noqa: E402
import numeric_bindings as numeric  # noqa: E402
import release_assets as assets  # noqa: E402

N_TEST, N_BOOT = 400, 100
I5 = str(mk.DEFAULT_I5)
DESIGNS = {
    'A': ('A', 'A', 'M+', {}),
    'B': ('B', 'B', 'M0', {}),
    'C': ('C', 'C', 'M-', {}),
    'I': ('A', 'A', 'M+', {'drop': [('envelope', 5678)]}),
    'mixed': ('A', 'B', 'M_partial', {'seeds': (1234, 5678, 9012), 'late': [('envelope', 9012)]}),
    'na': ('B', 'B', 'M_na', {}),
    # V6 regressions (producer-schema finals from synthetic predictions)
    'A3': ('A', 'A', 'M+', {'seeds': (1234, 5678, 9012)}),
    'noR1': ('A', 'A', 'M+', {'seeds': (1234, 5678, 9012), 'drop': [('random', 1234)]}),
    'noC1_late': ('A', 'A', 'M+', {'seeds': (1234, 5678, 9012), 'drop': [('centroid', 1234)],
                                    'cb_late': True}),
    'noC1_ontime': ('A', 'A', 'M+', {'seeds': (1234, 5678, 9012), 'drop': [('centroid', 1234)]}),
    'cb_seedB': ('A', 'A', 'M+', {'cb_seed': 5678}),
    'M0_negB': ('Asmall', 'Asmall', 'M0', {'cb_placement': 0.0020}),
    'Bbig_A': ('Bbig', 'A', 'M+', {}),
}
BANNED = ('comes from what remains visible', 'drives the gain', 'smaller',
          'comparable to run-to-run', 'points to the budget')


@pytest.fixture(scope='module')
def designs(tmp_path_factory):
    root = tmp_path_factory.mktemp('crr')
    return {key: fx.build_design(root / key, c, e, m, n_test=N_TEST, n_boot=N_BOOT, **kw)
            for key, (c, e, m, kw) in DESIGNS.items()}


def run_build(d, tmp, budgets=None, **kwargs):
    kwargs.setdefault('figure', False)
    return mk.build(d['final'], d['runs'], d['md'], budgets if budgets is not None else [I5],
                    out_dir=tmp / 'out', evidence_dir=tmp / 'evidence', allow_synthetic=True,
                    **kwargs)


def tex_macros(path):
    return {name: body for name, body, _, _ in assets.macros(open(path, encoding='utf-8').read())}


def prereg_label(deltas, random_values, n_registered_complete):
    """Independent restatement of PREREGISTRATION sections 3 and 8 (not cr_stats code)."""
    if n_registered_complete < 2 or len(random_values) != 1 + len(deltas):
        return 'incomplete'
    mean = sum(deltas) / len(deltas)
    if mean <= 0:
        return 'reversed'
    if all(d > 0 for d in deltas) and mean > max(random_values) - min(random_values):
        return 'consistent_direction'
    return 'within_run_to_run_variation'


# ---------------------------------------------------------------------------
# scenario selection
# ---------------------------------------------------------------------------

def test_scenario_selection_follows_the_preregistration(designs, tmp_path):
    expected = {'A': ('A', 'A', 'A', 'M+'), 'B': ('B', 'B', 'B', 'M0'),
                'C': ('C', 'C', 'C', 'M-'), 'I': ('I', 'I', 'I', 'M_incomplete'),
                'mixed': ('mixed', 'A', 'B', 'M_partial'), 'na': ('B', 'B', 'B', 'M_na')}
    for key, d in designs.items():
        if key not in expected:
            continue
        res = run_build(d, tmp_path / key)
        scen = res['scenario']
        assert (scen['scenario'], scen['codes']['centroid'], scen['codes']['envelope'],
                scen['matched_control']['label']) == expected[key], key
        final = json.load(open(d['final']))
        block = final['analysis'][fx.PRIMARY]
        auc = block['per_run_test_auc']
        ident = {n: e['identity'] for n, e in final['runs'].items()}
        name = {(i['arm'], i['train_seed'] if i['role'] == 'new' else None): n
                for n, i in ident.items()}
        complete = block['complete_seed_indices']
        randoms = [auc[name[('random', None)]]] + [auc[name[('random', s)]] for s in complete]
        for arm in ('centroid', 'envelope'):
            deltas = [auc[name[(arm, s)]] - auc[name[('random', s)]] for s in complete]
            label = prereg_label(deltas, randoms, len(complete))
            assert label == block['outcomes'][arm]['label'] == scen['labels'][arm], (key, arm)
            assert mk.LABEL_CODE[label] == scen['codes'][arm]
    # Amendment 1: the late seed-C ENVELOPE run makes index 9012 descriptive only.
    mixed = json.load(open(designs['mixed']['final']))['analysis'][fx.PRIMARY]
    assert mixed['complete_seed_indices'] == [1234, 5678]
    assert mixed['incomplete_seed_indices'] == [9012]


def _block(P=None, B=None, spread=0.003, flag=None, descriptive=False, p_desc=False,
           b_desc=False, seed=1234):
    contrasts, late = [], []
    if P is not None:
        (late if p_desc else contrasts).append({
                          'kind': 'centroid-random_cb', 'train_seed': seed, 'delta': P,
                          'ci95': [P - 0.01, P + 0.01], 'val_delta': P,
                          'abs_delta_exceeds_random_range': (abs(P) > spread if spread is not None
                                                             else None) if flag is None
                          else flag})
    if B is not None:
        (late if b_desc else contrasts).append({
            'kind': 'random_cb-random', 'train_seed': seed, 'delta': B,
            'ci95': [B - 0.01, B + 0.01], 'val_delta': B})
    realizations = [0.8600, 0.8600 + spread, 0.8610] if spread is not None else []
    outcomes = {arm: {'random_range': spread, 'random_realizations': realizations}
                for arm in ('centroid', 'envelope')}
    if descriptive:
        contrasts, late = [], contrasts + late
    return {'outcomes': outcomes, 'primary_contrasts': [], 'matched_control': contrasts,
            'descriptive_incomplete_contrasts': late, 'complete_seed_indices': [1234, 5678]}


ANCHOR = {'arm': 'random', 'role': 'anchor', 'seed': None, 'eligible': True}


def test_matched_control_rule_and_boundaries():
    """PREREGISTRATION amendment 2, labels only from on-time runs at seed 1234."""
    cb_run = {'cb': {'arm': 'random_cb', 'seed': 1234, 'eligible': True}, 'r0': ANCHOR}
    cases = [
        (dict(P=0.003, B=0.001), 'M0'),            # |P| == r -> not detected
        (dict(P=-0.003, B=0.004), 'M0'),
        (dict(P=0.0031, B=0.0031), 'M+'),          # P == (P + B) / 2 -> most of the gap
        (dict(P=0.0031, B=0.0032), 'M_partial'),   # P just under half of the same-seed gap
        (dict(P=0.0100, B=-0.002), 'M+'),
        (dict(P=-0.0031, B=0.009), 'M-'),
        (dict(P=-0.0031), 'M-'),                   # M- needs no B
        (dict(P=0.002), 'M0'),                     # M0 needs no B
        (dict(P=0.0050), 'M_unassessed'),          # V6-1: no B -> no M+
        (dict(P=0.0050, B=0.001, b_desc=True), 'M_unassessed'),  # late R1: B descriptive
        (dict(P=0.0050, B=0.001, descriptive=True), 'M_desc'),
        (dict(P=0.0050, B=0.001, p_desc=True), 'M_desc'),
        (dict(B=0.001), 'M_noP'),                  # V6-2: CB without a CENTROID comparator
        (dict(P=0.0050, B=0.001, spread=None), 'M_incomplete'),
    ]
    for kwargs, label in cases:
        info = mk.matched_outcome(_block(**kwargs), cb_run)
        assert info['label'] == label, kwargs
        assert (info['name'] is not None) == (label in mk.CONTROL_LABELS), kwargs
    assert mk.matched_outcome(_block(), {})['label'] == 'M_na'
    assert mk.matched_outcome(_block(), {})['reason'] == 'no RANDOM-CB result'
    assert mk.matched_outcome(_block(), cb_run)['label'] == 'M_noP'
    with pytest.raises(mk.InputError, match='disagrees'):
        mk.matched_outcome(_block(P=0.002, B=0.0, flag=True), cb_run)
    with pytest.raises(mk.InputError, match='seed 1234 only'):  # V6-1: unregistered CB seed
        mk.matched_outcome(_block(P=0.005, B=0.001, seed=5678),
                           {'cb': {'arm': 'random_cb', 'seed': 5678, 'eligible': True}})
    # V10-2: no registered label without the complete registered RANDOM reference set
    # (eligible original RANDOM + the new RANDOM runs of >= 2 complete indices); P and B stay.
    no_anchor = {'cb': cb_run['cb']}
    late_anchor = dict(cb_run, r0=dict(ANCHOR, eligible=False))
    one_index = dict(_block(P=0.0050, B=0.001), complete_seed_indices=[1234])
    for runs, block, why in ((no_anchor, _block(P=0.0050, B=0.001), ['anchor_missing']),
                             (late_anchor, _block(P=0.0050, B=0.001), ['anchor_late']),
                             (cb_run, one_index, ['indices']),
                             (no_anchor, _block(P=0.0020, B=0.001), ['anchor_missing'])):
        info = mk.matched_outcome(block, runs)
        assert info['label'] == 'M_incomplete' and info['name'] is None, why
        assert info['reference_missing'] == why and info['spread'] is None
        assert info['P'] is not None and info['B'] is not None


def test_scenario_mapping_is_the_final_labels_unchanged():
    for c in mk.LABEL_CODE:
        for e in mk.LABEL_CODE:
            scen, codes = mk.scenario_of({'centroid': c, 'envelope': e})
            assert codes == {'centroid': mk.LABEL_CODE[c], 'envelope': mk.LABEL_CODE[e]}
            assert scen == (codes['centroid'] if codes['centroid'] == codes['envelope']
                            else 'mixed')


# ---------------------------------------------------------------------------
# numbers come from the inputs
# ---------------------------------------------------------------------------

def test_numbers_are_formatted_from_the_inputs(designs, tmp_path):
    d = designs['mixed']
    res = run_build(d, tmp_path, budgets=[I5, d['cb_audit']])
    out = tmp_path / 'out'
    macros = tex_macros(out / 'cr_results.tex')
    final = json.load(open(d['final']))
    md = json.load(open(d['md']))
    block = final['analysis'][fx.PRIMARY]
    by_slot = {(e['identity']['arm'], e['identity']['train_seed']): n
               for n, e in final['runs'].items()}
    c1, r1 = by_slot[('centroid', 1234)], by_slot[('random', 1234)]
    assert macros['RepAUCCentroidSeedA'] == '%.4f' % block['per_run_test_auc'][c1]
    assert macros['RepValAUCCentroidSeedA'] == '%.4f' % block['per_run_val_auc'][c1]
    run_dir = [r for r in d['runs'] if os.path.basename(r) == c1][0]
    receipt = json.load(open(os.path.join(run_dir, 'unsealed_results.json')))
    assert macros['RepHeadSDCentroidSeedA'] == '%.4f' % receipt['variants'][fx.PRIMARY]['test_auc_sd']
    outcome = block['outcomes']['centroid']
    assert macros['RepDeltaMeanCentroid'] == '%+.4f' % outcome['mean_delta']
    assert macros['RepRandomRange'] == '%.4f' % outcome['random_range']
    contrast = [c for c in block['primary_contrasts']
                if c['kind'] == 'centroid-random' and c['train_seed'] == 1234][0]
    assert macros['RepDeltaCentroidSeedA'] == '%+.4f' % contrast['delta']
    assert macros['RepDeltaCentroidSeedACI'] == r'[%+.4f,\,%+.4f]' % tuple(contrast['ci95'])
    # Seed C is incomplete (late ENVELOPE): its CENTROID contrast is descriptive, not primary.
    late = [c for c in block['descriptive_incomplete_contrasts'] if c['train_seed'] == 9012]
    assert late and macros['RepDeltaCentroidSeedC'] == '%+.4f' % late[0]['delta']
    placement = [c for c in block['matched_control'] if c['kind'] == 'centroid-random_cb'][0]
    assert macros['RepPlacementDelta'] == '%+.4f' % placement['delta']
    assert macros['RepNBoot'].replace('{,}', '') == str(final['n_boot'])
    mean_r = np.mean([block['per_run_test_auc'][by_slot[('random', s)]] for s in (1234, 5678)])
    assert macros['RepMeanAUCRandom'] == '%.4f' % mean_r
    md_c1 = md['runs'][c1]['subsets']['glaucoma']['test']['ridge']
    assert macros['RepMDMAECentroidSeedA'] == '%.2f' % md_c1['mae']
    assert macros['RepBudgetURandomCB'] == '%.1f' % 102.6  # cb audit fixture row
    # Every numeric macro re-evaluates from the staged byte copies (as p15 would).
    evidence = numeric.Evidence(mk.PAPER, tmp_path / 'evidence',
                                {n: {'root': 'stage', 'path': os.path.relpath(
                                    spec['staged'], tmp_path / 'evidence').replace('\\', '/'),
                                     'sha256': spec['sha256']}
                                 for n, spec in res['inputs'].items()})
    evidence.roots['stage'] = tmp_path / 'evidence'
    if res['p1c_stats']:
        evidence.roots['stats'] = mk.DEFAULT_STATS
    for record in res['macros']:
        expected = evidence.binding(record['expression'])['expected']
        assert macros[record['name']].replace('{,}', '') == expected.replace('{,}', '')
        assert re.search(r'\d', macros[record['name']])
        sources = set(record['source_hashes'])
        assert sources <= set(res['inputs']) | {'p1c_stats.json'}
    for name, spec in res['inputs'].items():
        assert mk.sha256_file(spec['staged']) == mk.sha256_file(spec['original']) == spec['sha256']


def test_figure_is_headless_and_bound_to_inputs(designs, tmp_path):
    import matplotlib
    d = designs['A']
    res = run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    evidence = json.load(open(out / 'fig_cr_seed_spread.evidence.json'))
    assert matplotlib.get_backend().lower() == 'agg'
    assert evidence['outputs']['png'] == mk.sha256_file(out / 'fig_cr_seed_spread.png')
    assert evidence['outputs']['pdf'] == mk.sha256_file(out / 'fig_cr_seed_spread.pdf')
    assert len(evidence['points']) == len(d['runs'])
    for name, spec in evidence['inputs'].items():
        assert spec['sha256'] == res['inputs'][name]['sha256']
    final = json.load(open(d['final']))
    for point in evidence['points']:
        assert point['test_auc'] == final['analysis'][fx.PRIMARY]['per_run_test_auc'][point['run']]
    figure_tex = open(out / 'cr_figure_seed_spread.tex').read()
    caption = tex_macros(out / 'cr_results.tex')['RepFigSeedSpreadCaption']
    # I12: the caption text itself is in the float, so the p15 figure receipt hashes it.
    assert figure_tex.count(r'\caption{%s}' % caption) == 1 and not re.search(r'\d', caption)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _copy_design(d, dst):
    shutil.copytree(d['root'], dst)
    runs = [os.path.join(str(dst), 'probes', os.path.basename(r)) for r in d['runs']]
    return dict(d, root=str(dst), runs=runs, final=os.path.join(str(dst), 'final.json'),
                md=os.path.join(str(dst), 'md_test.json'))


def _edit_json(path, edit):
    data = json.load(open(path))
    edit(data)
    with open(path, 'w') as stream:
        json.dump(data, stream)


def test_refuses_inputs_without_unseal_receipt_or_with_incomplete_identity(designs, tmp_path):
    base = designs['A']
    case = lambda name: _copy_design(base, tmp_path / name)  # noqa: E731
    target = lambda d, arm='centroid_s1234': [r for r in d['runs']  # noqa: E731
                                              if os.path.basename(r) == arm][0]
    d = case('no_receipt')
    os.remove(os.path.join(target(d), 'unsealed_results.json'))
    with pytest.raises(mk.InputError, match='not been unsealed'):
        run_build(d, tmp_path / 'o1')
    d = case('foreign_receipt')
    _edit_json(os.path.join(target(d), 'unsealed_results.json'),
               lambda r: r.update(sealed_manifest_sha256='0' * 64))
    with pytest.raises(mk.InputError, match='not bound'):
        run_build(d, tmp_path / 'o2')
    for key in ('checkpoint_sha256', 'train_seed', 'run_uuid', 'epoch'):
        d = case('no_' + key)
        _edit_json(os.path.join(target(d), 'results.json'),
                   lambda r: r['identity'].pop(key))
        with pytest.raises(mk.InputError, match='incomplete or unregistered identity'):
            run_build(d, tmp_path / ('o_' + key))
    d = case('final_identity')
    _edit_json(d['final'], lambda f: f['runs']['centroid_s1234']['identity'].update(
        train_seed=5678))
    with pytest.raises(mk.InputError, match='final JSON identity differs'):
        run_build(d, tmp_path / 'o3')
    d = case('final_auc')
    _edit_json(d['final'], lambda f: f['analysis'][fx.PRIMARY]['per_run_test_auc'].update(
        centroid_s1234=f['analysis'][fx.PRIMARY]['per_run_test_auc']['centroid_s1234'] + 1e-3))
    with pytest.raises(mk.InputError, match='not the unsealed head-seed mean'):
        run_build(d, tmp_path / 'o4')
    d = case('missing_dir')
    with pytest.raises(mk.InputError, match='without a probe directory'):
        mk.build(d['final'], d['runs'][1:], None, [I5], out_dir=tmp_path / 'o5',
                 evidence_dir=tmp_path / 'e5', allow_synthetic=True, figure=False)
    # The fixtures are pre-freeze: without --allow-synthetic they are refused.
    with pytest.raises(mk.InputError, match='before the results freeze'):
        mk.build(d['final'], d['runs'], None, [I5], out_dir=tmp_path / 'o6',
                 evidence_dir=tmp_path / 'e6', figure=False)
    # MD must be the test split.
    d = case('md_validation')
    _edit_json(d['md'], lambda m: m.update(split='Validation'))
    with pytest.raises(mk.InputError, match='test split only'):
        run_build(d, tmp_path / 'o7')


def test_synthetic_outputs_never_reach_the_paper(designs, tmp_path):
    d = designs['A']
    before = {p: mk.sha256_file(p) for p in mk.DEFAULT_OUT.glob('*')} if mk.DEFAULT_OUT.is_dir() else {}
    with pytest.raises(mk.InputError, match='outside the paper'):
        mk.build(d['final'], d['runs'], d['md'], [I5], out_dir=mk.DEFAULT_OUT,
                 evidence_dir=tmp_path / 'e', allow_synthetic=True, figure=False)
    with pytest.raises(mk.InputError, match='outside the paper'):
        mk.build(d['final'], d['runs'], d['md'], [I5], out_dir=tmp_path / 'o',
                 evidence_dir=mk.DEFAULT_EVIDENCE, allow_synthetic=True, figure=False)
    with pytest.raises(mk.InputError, match='real inputs'):
        mk.build(d['final'], d['runs'], d['md'], [I5], out_dir=tmp_path / 'o',
                 evidence_dir=tmp_path / 'e', allow_synthetic=True, update_reviews=True)
    after = {p: mk.sha256_file(p) for p in mk.DEFAULT_OUT.glob('*')} if mk.DEFAULT_OUT.is_dir() else {}
    assert before == after
    staged = mk.DEFAULT_EVIDENCE / 'final.json'
    assert not staged.exists() or json.load(open(staged)).get('before_freeze') is False


# ---------------------------------------------------------------------------
# no hand-typed numbers
# ---------------------------------------------------------------------------

def test_no_hand_typed_numbers_in_generated_text_or_tables(designs, tmp_path):
    for key, d in designs.items():
        if key == 'cb_seedB':
            continue
        run_build(d, tmp_path / key, figure=False)
        out = tmp_path / key / 'out'
        macros = tex_macros(out / 'cr_results.tex')
        numeric_names = {m for m, body in macros.items() if re.search(r'\d', body)}
        text_names = set(macros) - numeric_names
        for name in text_names:
            assert not re.search(r'\d', macros[name]), (key, name)
        for name in numeric_names:  # numeric macros hold one display value, no prose
            assert not re.search(r'[A-Za-z]{3,}', macros[name].replace(r'\,', '')), (key, name)
        defined = set(macros)
        for path in sorted(out.glob('cr_*.tex')):
            source = assets.uncomment(path.read_text(encoding='utf-8'))
            rel = 'auto/' + path.name
            rows = list(numeric.literals(source, rel))
            loose = [r for r in rows if r['status'] != 'structural']
            assert not loose, (key, path.name, [(r['value'], r['context'][:60]) for r in loose])
            used = set(re.findall(r'\\(Rep[A-Za-z]+)', source))
            assert used <= defined, (key, path.name, used - defined)
        tldr = (out / 'cr_tldr.txt').read_text().strip()
        assert len(tldr) <= mk.TLDR_LIMIT and not re.search(r'\d', tldr)


def test_every_scenario_text_is_digit_free_and_complete(designs):
    d = designs['mixed']
    ctx, _, _ = mk.prepare(d['final'], d['runs'], d['md'], [I5, d['cb_audit']],
                           evidence_dir=os.path.join(d['root'], '_ev'), allow_synthetic=True)
    labels = list(mk.LABEL_CODE)
    for c in labels:
        for e in labels:
            for m in ('M+', 'M0', 'M-', 'M_partial', 'M_desc', 'M_na', 'M_incomplete',
                      'M_unassessed', 'M_noP'):
                ctx.labels = {'centroid': c, 'envelope': e}
                ctx.scenario, ctx.codes = mk.scenario_of(ctx.labels)
                ctx.matched = dict(ctx.matched, label=m)
                text = mk.build_text(ctx)
                for name, body in text.macros.items():
                    assert not re.search(r'\d', body), (c, e, m, name)
                tldr = mk.tldr_text(ctx)
                assert len(tldr) <= mk.TLDR_LIMIT
                prose = ' '.join(list(text.macros.values()) + [tldr])
                assert not [b for b in BANNED if b in prose], (c, e, m)
                refs = set(re.findall(r'\\(Rep[A-Za-z]+)', ' '.join(text.macros.values())))
                assert refs <= set(ctx.values) | set(text.macros), (c, e, m)
    with pytest.raises(ValueError, match='digits'):
        mk.Text(ctx).put('RepBad', 'an AUC of 0.87')


# ---------------------------------------------------------------------------
# check mode, reviews, integration, CLI
# ---------------------------------------------------------------------------

def test_cb_audit_from_trainer_log_and_budget_precedence(designs, tmp_path):
    train_patch = pytest.importorskip('src.train_patch')
    base = dict(matching='strict', bypass_ramp0=0, target_len=40, context=81, loss_slots=160,
                unique_targets=102.5, duplicates=57.5, hidden_shadow=112.8, hidden_matched=112.8,
                proposals_mean=4000.0, proposals_max=20000, exact_fallbacks=0, shadow_ms=80.0,
                match_ms=40.0)
    lines = ['noise line',
             train_patch.format_cb_stats(dict(base, r_t=0.4, unique_targets=200.0)),  # ramp
             train_patch.format_cb_stats(dict(base, r_t=1.0)),
             train_patch.format_cb_stats(dict(base, r_t=1.0, context=79, unique_targets=104.5,
                                              duplicates=55.5)),
             train_patch.format_cb_stats(dict(base, r_t=0.0, bypass_ramp0=1))]
    log = tmp_path / 'train_cb.log'
    log.write_text('\n'.join(lines) + '\n')
    audit = mk.cb_audit_from_log([log])
    assert audit['batches'] == 2 and mk.budget_kind(audit) == 'cb'
    row = audit['rows']['random_cb']
    assert row['unique_targets_U'] == pytest.approx(103.5)
    assert row['context_tokens'] == pytest.approx(80.0)
    assert row['duplicate_slots_D'] == pytest.approx(56.5) and row['loss_slots_L'] == 160
    assert audit['mean_abs_hidden_diff'] == 0
    out = tmp_path / 'cb_audit.json'
    assert mk.main(['cb-audit', '--log', str(log), '--out', str(out)]) == 0
    empty = tmp_path / 'empty.log'
    empty.write_text('nothing\n')
    with pytest.raises(mk.InputError, match='no full-ramp'):
        mk.cb_audit_from_log([empty])
    bench = tmp_path / 'bench.json'
    bench.write_text(json.dumps({'results': {'strict_ep30': {
        'images': 960, 'per_image_mismatches_vs_shadow': {'U': 0, 'L': 0, 'D': 0, 'C': 0, 'H': 0},
        'purity': {'centroid': {'target_purity_pct': 45.95}, 'cb': {'target_purity_pct': 36.72},
                   'random': {'target_purity_pct': 35.97}, 'n_valid_views_per_draw': 190}}}}))
    run_build(designs['A'], tmp_path / 'b', budgets=[str(out), I5, str(bench)])
    macros = tex_macros(tmp_path / 'b' / 'out' / 'cr_results.tex')
    assert macros['RepBudgetURandomCB'] == '103.5' and macros['RepCBBatches'] == '2'
    i5 = json.load(open(I5))['summary']
    assert macros['RepBudgetURandom'] == '%.1f' % i5['random']['unique_targets_U']['mean']
    assert macros['RepPurityRandomCB'] == '36.7' and macros['RepCBMismatches'] == '0'
    table = (tmp_path / 'b' / 'out' / 'cr_table_control.tex').read_text()
    assert r'\RepBudgetURandomCB' in table and "run's mask statistics" in table


def test_check_mode_detects_stale_outputs(designs, tmp_path):
    d = designs['B']
    run_build(d, tmp_path)
    assert run_build(d, tmp_path, check=True)['stale'] == []
    table = tmp_path / 'out' / 'cr_table_replication.tex'
    table.write_text(table.read_text() + '% edited\n')
    assert run_build(d, tmp_path, check=True)['stale'] == ['cr_table_replication.tex']
    staged = tmp_path / 'evidence' / 'final.json'
    staged.write_bytes(staged.read_bytes() + b' ')
    with pytest.raises(mk.InputError, match='stale'):
        run_build(d, tmp_path, check=True)


def test_update_reviews_owns_only_rep_entries(tmp_path):
    reviews = tmp_path / 'numeric_reviews.json'
    shutil.copyfile(mk.REVIEWS, reviews)
    original = json.load(open(reviews, encoding='utf-8'))

    class FakeStage(object):
        sources = {'rep_final': {'root': 'repo', 'path': 'x/final.json', 'sha256': 'a' * 64}}

    records = [{'name': 'RepAUCRandomSeedA',
                'expression': numeric.fmt('%.4f', numeric.ref('rep_final', 'n_boot'))}]
    mk.write_reviews(FakeStage(), records, path=reviews)
    data = json.load(open(reviews, encoding='utf-8'))
    assert data['macros']['RepAUCRandomSeedA'] == {'expression': records[0]['expression']}
    assert data['sources']['rep_final']['sha256'] == 'a' * 64
    for name, spec in original['macros'].items():
        assert data['macros'][name] == spec
    for name, spec in original['sources'].items():
        assert data['sources'][name] == spec
    FakeStage.sources = {'rep_final': {'root': 'stage', 'path': 'final.json', 'sha256': 'a' * 64}}
    with pytest.raises(mk.InputError, match='repository'):
        mk.write_reviews(FakeStage(), records, path=reviews)


def test_integration_replaces_every_placeholder(designs, tmp_path):
    d = designs['mixed']
    run_build(d, tmp_path, figure=True)
    paper = tmp_path / 'paper'
    (paper / 'auto').mkdir(parents=True)
    for name in ('main_submission.tex', 'compact_protocol.tex'):
        shutil.copyfile(mk.PAPER / name, paper / name)
    crlf = b'\r\n' in (paper / 'main_submission.tex').read_bytes()
    mk.integrate(paper, tmp_path / 'out')
    main = (paper / 'main_submission.tex').read_bytes().decode('utf-8')
    appendix = (paper / 'compact_protocol.tex').read_text(encoding='utf-8')
    for token in (r'\CRSeedResultsPending', r'\CRControlResultsPending',
                  r'\CRPoolingResultsPending', r'\CRCell', r'\label{tab:replication}'):
        assert token not in main
    for token in (r'\input{auto/cr_results}', r'\RepAbstractResults', r'\RepIntroResults',
                  r'\RepContribResults', r'\RepContribControlItem', r'\RepSetupResults',
                  r'\RepResultsSeeds', r'\RepResultsControl', r'\RepResultsPooling',
                  r'\RepDiscussionLead', r'\RepLimitations', r'\RepConclusion',
                  r'\input{auto/cr_table_replication}', r'\input{auto/cr_table_control}'):
        assert main.count(token) == 1, token
    assert ('\r\n' in main) == crlf
    assert r'\RepAppendixResults' in appendix and r'\input{auto/cr_table_md}' in appendix
    assert (paper / 'auto' / 'fig_cr_seed_spread.png').is_file()
    with pytest.raises(mk.InputError, match='expected one occurrence'):
        mk.integrate(paper, tmp_path / 'out')  # already integrated: refuses a second pass
    before = mk.sha256_file(mk.PAPER / 'main_submission.tex')
    with pytest.raises(mk.InputError, match='real paper'):
        mk.integrate(mk.PAPER, tmp_path / 'out', allow_real_paper=True)
    assert mk.sha256_file(mk.PAPER / 'main_submission.tex') == before


def test_cli_build_and_refusal(designs, tmp_path):
    d = designs['C']
    args = ['build', '--final', d['final'], '--inventory', d['inventory'], '--md', d['md'],
            '--budget-i5', I5, '--out-dir', str(tmp_path / 'out'),
            '--evidence-dir', str(tmp_path / 'ev'), '--no-figure']
    assert mk.main(args) == 2  # pre-freeze fixtures without --allow-synthetic
    assert mk.main(args + ['--allow-synthetic']) == 0
    assert mk.main(args + ['--allow-synthetic', '--check']) == 0
    assert (tmp_path / 'out' / 'cr_results.values.json').is_file()


def test_real_md_producer_schema(tmp_path):
    """cr_md_regression --split test output (real producer) is accepted and bound."""
    torch = pytest.importorskip('torch')  # noqa: F841
    import test_cr_eval_md as md_tests
    import cr_md_regression as md_reg
    rng = np.random.default_rng(11)
    root = tmp_path / 'md'
    data_dir = root / 'data'
    cases, start = {}, 1
    for split, n in md_tests.N.items():
        (data_dir / split).mkdir(parents=True)
        names = ['data_%05d.npz' % (start + i) for i in range(n)]
        start += n
        glaucoma = np.arange(n) % 2 == 1
        md = np.where(glaucoma, rng.normal(-7, 4, n), rng.normal(0, 1.5, n)).round(2)
        cases[split] = {'names': names, 'glaucoma': glaucoma, 'md': md}
    lines = ['filename,age,md,glaucoma,use']
    for split, c in cases.items():
        for name, g, m in zip(c['names'], c['glaucoma'], c['md']):
            lines.append('%s,50,%s,%s,%s' % (name, m, 'yes' if g else 'no', md_tests.USE[split]))
    meta = root / 'meta.csv'
    meta.write_text('\n'.join(lines) + '\n')
    cohort = {'root': root, 'data_dir': data_dir, 'cases': cases, 'meta': str(meta),
              'weights': rng.normal(0, 1, md_tests.D)}
    runs = [md_tests.make_md_run(cohort, 'orig_random', 'random', None, 'anchor', 1.0, legacy=True),
            md_tests.make_md_run(cohort, 'r_1234', 'random', 1234, 'new', 1.0),
            md_tests.make_md_run(cohort, 'c_1234', 'centroid', 1234, 'new', 0.3),
            md_tests.make_md_run(cohort, 'e_1234', 'envelope', 1234, 'new', 3.0)]
    for run in runs:
        _edit_json(os.path.join(run, 'results.json'), lambda r: r['variants'][fx.PRIMARY].update(
            val_auc_mean=0.8))
        cr_stats.cmd_unseal(run, allow_before_freeze=True)
    final = cr_stats.cmd_final(runs, n_boot=20, out=str(tmp_path / 'final.json'),
                               allow_before_freeze=True, head_seeds=(42,))
    md_reg.cmd_md(runs, 'test', metadata=cohort['meta'], n_boot=20,
                  out=str(tmp_path / 'md.json'), allow_before_freeze=True)
    res = mk.build(str(tmp_path / 'final.json'), runs, str(tmp_path / 'md.json'), [I5],
                   out_dir=tmp_path / 'out', evidence_dir=tmp_path / 'ev', allow_synthetic=True,
                   figure=False)
    macros = tex_macros(tmp_path / 'out' / 'cr_results.tex')
    md = json.load(open(tmp_path / 'md.json'))
    ridge = md['runs']['c_1234']['subsets']['glaucoma']['test']['ridge']
    assert macros['RepMDMAECentroidSeedA'] == '%.2f' % ridge['mae']
    assert 'RepMDMAERandomOrig' not in macros  # legacy anchor excluded by the producer
    assert res['scenario']['scenario'] == 'I'  # one seed index: no registered label
    assert final['analysis'][fx.PRIMARY]['outcomes']['centroid']['label'] == 'incomplete'
    caption = (tmp_path / 'out' / 'cr_table_md.tex').read_text()
    assert 'not included' in caption


# ---------------------------------------------------------------------------
# V6 regressions
# ---------------------------------------------------------------------------

def _texts(res):
    return {t['name']: t['text'] for t in res['text_macros']}


def test_v6_partial_controls_and_registered_seed(designs, tmp_path):
    # V6-1: R1 missing -> P > r but no B -> no M+ / "most of the gap" claim.
    res = run_build(designs['noR1'], tmp_path / 'r1')
    info = res['scenario']['matched_control']
    assert info['label'] == 'M_unassessed' and info['B'] is None and info['P'] > info['spread']
    texts = _texts(res)
    claims = ' '.join(v for k, v in texts.items() if k != 'RepAppendixRule')  # rule definition
    assert 'most of the' not in claims
    assert 'budget shift is not available' in texts['RepResultsControl']
    assert texts['RepContribControlItem'] == ''
    # V6-2: C1 missing with a late CB -> no crash, nothing called uncompleted that finished.
    res = run_build(designs['noC1_late'], tmp_path / 'c1late')
    assert res['scenario']['matched_control']['label'] == 'M_noP'
    ctrl = _texts(res)['RepResultsControl']
    assert 'finished after the results freeze' in ctrl
    assert 'placement difference is not available' in ctrl and '(descriptive)' in ctrl
    assert 'did not produce' not in ctrl
    # ... and with an on-time CB: the B result is reported, the control is not "uncompleted".
    res = run_build(designs['noC1_ontime'], tmp_path / 'c1')
    ctrl = _texts(res)['RepResultsControl']
    assert 'placement difference is not available' in ctrl and r'\RepBudgetShift' in ctrl
    assert 'did not produce' not in ctrl and 'finished after' not in ctrl
    table = (tmp_path / 'c1' / 'out' / 'cr_table_control.tex').read_text()
    assert r'\RepAUCRandomCBSeedA' in table
    # V6-1: RANDOM-CB is registered at seed 1234 only.
    with pytest.raises(mk.InputError, match='seed 1234 only'):
        run_build(designs['cb_seedB'], tmp_path / 'cbB')


def test_v6_prose_is_literal_for_its_label(designs, tmp_path):
    res = run_build(designs['M0_negB'], tmp_path / 'm0')
    info = res['scenario']['matched_control']
    assert res['scenario']['scenario'] == 'A' and info['label'] == 'M0' and info['B'] < 0
    prose = ' '.join(_texts(res).values()) + ' ' + res['tldr']
    assert not [b for b in BANNED if b in prose]
    assert 'within run-to-run variation' in _texts(res)['RepAbstractResults']
    res = run_build(designs['Bbig_A'], tmp_path / 'bb')
    scen = res['scenario']
    assert scen['scenario'] == 'mixed' and scen['codes'] == {'centroid': 'B', 'envelope': 'A'}
    outcomes = json.load(open(designs['Bbig_A']['final']))['analysis'][fx.PRIMARY]['outcomes']
    assert outcomes['centroid']['mean_delta'] > outcomes['envelope']['mean_delta']  # V6 case
    texts = _texts(res)
    prose = ' '.join(texts.values()) + ' ' + res['tldr']
    assert not [b for b in BANNED if b in prose]
    assert 'on average but not in every continuation' in texts['RepAbstractResults']


def _paper_copy(dst):
    (dst / 'auto').mkdir(parents=True)
    for name in ('main_submission.tex', 'compact_protocol.tex'):
        shutil.copyfile(mk.PAPER / name, dst / name)
    return dst


def test_v6_methods_text_follows_the_completed_indices(designs, tmp_path):
    for key, words in (('A3', ('three', 'all three', 'four')), ('A', ('two', 'both', 'three')),
                       ('I', ('one', None, 'two'))):
        run_build(designs[key], tmp_path / key, figure=True)
        out = tmp_path / key / 'out'
        macros = tex_macros(out / 'cr_results.tex')
        assert macros['RepNSeedsWord'] == words[0] and macros['RepNRandomWord'] == words[2]
        assert macros.get('RepAllSeedsWords') == words[1]
        assert macros['RepNRegisteredSeedsWord'] == 'three'
        paper = _paper_copy(tmp_path / key / 'paper')
        mk.integrate(paper, out)
        text = ' '.join(re.sub(r'\s+', ' ', (paper / n).read_text(encoding='utf-8'))
                        for n in ('main_submission.tex', 'compact_protocol.tex'))
        assert not [p for p in mk.OBSOLETE_PHRASES if p in text], key
        assert r'\RepAppendixRule{}' in text
        assert r'It adds \RepNRegisteredSeedsWord{} further continuations' in text
        rule = macros['RepAppendixRule']
        assert (r'\RepAllSeedsWords{}' in rule) == (words[1] is not None)
        assert 'registered amendment' in macros['RepSetupResults']


def test_v6_integration_metadata_fails_closed(designs, tmp_path, monkeypatch):
    d = designs['A']
    run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    sidecar = out / 'cr_results.values.json'
    original = sidecar.read_bytes()
    data = json.loads(original)
    no_flag = {k: v for k, v in data.items() if k != 'synthetic'}
    for i, (payload, match) in enumerate((({}, 'unknown schema'),
                                          (no_flag, 'explicit synthetic flag'),
                                          (dict(data, synthetic='false'), 'explicit synthetic'),
                                          (dict(data, synthetic=False), 'synthetic marker'))):
        sidecar.write_text(json.dumps(payload))
        with pytest.raises(mk.InputError, match=match):
            mk.integrate(_paper_copy(tmp_path / ('p%d' % i)), out)
    sidecar.write_bytes(original)
    table = out / 'cr_table_control.tex'
    good = table.read_bytes()
    table.write_bytes(good + b'% edited\n')
    with pytest.raises(mk.InputError, match='recorded hash'):
        mk.integrate(_paper_copy(tmp_path / 'p_tamper'), out)
    table.write_bytes(good)
    surrogate = _paper_copy(tmp_path / 'surrogate_real')
    before = mk.sha256_file(surrogate / 'main_submission.tex')
    monkeypatch.setattr(mk, 'PAPER', surrogate)
    with pytest.raises(mk.InputError, match='real paper'):
        mk.integrate(surrogate, out, allow_real_paper=True)
    sidecar.write_text(json.dumps(dict(data, synthetic=False)))
    with pytest.raises(mk.InputError, match='synthetic marker'):
        mk.integrate(surrogate, out, allow_real_paper=True)
    sidecar.write_bytes(original)
    assert mk.sha256_file(surrogate / 'main_submission.tex') == before
    monkeypatch.undo()
    mk.integrate(_paper_copy(tmp_path / 'p_ok'), out)  # intact synthetic set into a scratch copy
    # Without a figure, the figure float and its reference are not inserted.
    mk.build(d['final'], d['runs'], d['md'], [I5], out_dir=tmp_path / 'nofig',
             evidence_dir=tmp_path / 'ev2', allow_synthetic=True, figure=False)
    paper = _paper_copy(tmp_path / 'p_nofig')
    mk.integrate(paper, tmp_path / 'nofig')
    appendix = (paper / 'compact_protocol.tex').read_text(encoding='utf-8')
    assert 'cr_figure_seed_spread' not in appendix
    assert 'fig:seed_spread' not in tex_macros(tmp_path / 'nofig' / 'cr_results.tex')[
        'RepAppendixResults']


def test_v6_check_detects_corrupted_or_missing_figure(designs, tmp_path):
    d = designs['B']
    run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    assert run_build(d, tmp_path, check=True, figure=True)['stale'] == []
    png = out / 'fig_cr_seed_spread.png'
    good = png.read_bytes()
    png.write_bytes(b'not a png')
    assert 'fig_cr_seed_spread.png' in run_build(d, tmp_path, check=True, figure=True)['stale']
    png.write_bytes(good)
    evidence = out / 'fig_cr_seed_spread.evidence.json'
    ev_good = evidence.read_bytes()
    _edit_json(evidence, lambda e: e['points'][0].update(test_auc=0.5))
    assert 'fig_cr_seed_spread.evidence.json' in run_build(d, tmp_path, check=True,
                                                           figure=True)['stale']
    evidence.write_bytes(ev_good)
    (out / 'fig_cr_seed_spread.pdf').unlink()
    assert 'fig_cr_seed_spread.pdf' in run_build(d, tmp_path, check=True, figure=True)['stale']
    (out / 'cr_results.values.json').unlink()
    assert 'cr_results.values.json' in run_build(d, tmp_path, check=True, figure=True)['stale']


def test_v6_check_detects_corrupted_provenance(designs, tmp_path):
    """V6 re-check P2: provenance-only corruption of the figure evidence or the sidecar."""
    d = designs['B']
    run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    check = lambda: run_build(d, tmp_path, check=True, figure=True)['stale']  # noqa: E731
    assert check() == []
    evidence, sidecar = out / 'fig_cr_seed_spread.evidence.json', out / 'cr_results.values.json'
    for path, edit, name in (
            (evidence, lambda e: e['inputs']['rep_final'].update(sha256='0' * 64), evidence.name),
            (evidence, lambda e: e.update(script_sha256='0' * 64), evidence.name),
            (evidence, lambda e: e.update(schema='other'), evidence.name),
            (evidence, lambda e: e['inputs'].pop(sorted(e['inputs'])[-1]), evidence.name),
            (sidecar, lambda e: e['generator'].update(sha256='0' * 64), sidecar.name),
            (sidecar, lambda e: e['inputs']['rep_final'].update(sha256='0' * 64), sidecar.name)):
        good = path.read_bytes()
        _edit_json(path, edit)
        assert name in check(), name
        path.write_bytes(good)
    assert check() == []


# ---------------------------------------------------------------------------
# I12: post-freeze runbook automation (coordinator notes 2-4, TL;DR coverage, Word gate)
# ---------------------------------------------------------------------------

def test_i12_integration_passages_match_the_current_paper():
    """Every passage integrate replaces still occurs exactly once, applied in order, in the
    current paper source (a trimmed sentence once broke the discussion passage). The list is
    integration_pairs, which cr_migrate_receipts.py also uses."""
    for scenario, figure in (('A', True), ('B', True), ('mixed', False)):
        texts = {rel: (mk.PAPER / rel).read_text(encoding='utf-8').replace('\r\n', '\n')
                 for rel in ('main_submission.tex', 'compact_protocol.tex')}
        pairs = mk.integration_pairs(scenario, figure)
        assert (('main_submission.tex', mk.RANKING_SENTENCE, mk.RANKING_SINGLE_RUN, False)
                in pairs) == (scenario != 'A')
        for rel, old, new, drop in pairs:
            texts[rel] = mk._replace_once(rel, texts[rel], old, new, drop_newline=drop)
        assert len(mk.REPLICATION_TABLE.findall(texts['main_submission.tex'])) == 1
    assert not re.search(r'\d', mk.RANKING_SINGLE_RUN + mk.FIG3B_SEED_SPREAD)


def test_i12_note2_edits_follow_the_scenario(designs, tmp_path):
    flat = lambda text: re.sub(r'\s+', ' ', text)  # noqa: E731
    for key, keep_ranking, figure in (('A', True, True), ('B', False, True),
                                      ('mixed', False, False), ('I', False, True)):
        run_build(designs[key], tmp_path / key, figure=figure)
        paper = _paper_copy(tmp_path / key / 'paper')
        mk.integrate(paper, tmp_path / key / 'out')
        main = (paper / 'main_submission.tex').read_text(encoding='utf-8')
        assert '% CR:' not in main, key
        assert (flat(mk.RANKING_SENTENCE) in flat(main)) == keep_ranking, key
        assert (flat(mk.RANKING_SINGLE_RUN) in flat(main)) == (not keep_ranking), key
        spread = r'Figure~\ref{fig:seed_spread}' if figure else r'Table~\ref{tab:replication_runs}'
        assert flat(main).count(flat(mk.FIG3B_SEED_SPREAD % spread)) == 1, key
    sidecar = tmp_path / 'B' / 'out' / 'cr_results.values.json'
    _edit_json(sidecar, lambda s: s['scenario'].update(scenario='A'))
    with pytest.raises(mk.InputError, match='disagrees'):
        mk.integrate(_paper_copy(tmp_path / 'tampered'), tmp_path / 'B' / 'out')


def _mini_repo(tmp, d):
    """Stand-in repository (as the rehearsal mirror): registry and review file copies, every
    .tex input of the paper, synthetic results in paper/auto, evidence inside the repo."""
    import cr_postfreeze as pf
    repo = tmp / 'repo'
    paper = repo / 'paper' / 'genai4health2026'
    for src in [mk.PAPER / n for n in ('numeric_reviews.json',)] + \
            sorted(mk.PAPER.glob('*.tex')) + sorted((mk.PAPER / 'auto').glob('*.tex')):
        dst = paper / src.relative_to(mk.PAPER)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    (repo / 'autopilot').mkdir()
    shutil.copyfile(mk.REGISTRY, repo / 'autopilot' / 'release_assets.json')
    (repo / pf.MIRROR_MARKER).write_text('{"note": "test stand-in"}')
    res = mk.build(d['final'], d['runs'], d['md'], [I5], out_dir=paper / 'auto',
                   evidence_dir=repo / 'autopilot' / 'investigations' / 'results',
                   allow_synthetic=True, figure=True)
    return repo, paper, res


def test_i12_register_figure_writes_producer_and_receipt(designs, tmp_path):
    import cr_postfreeze as pf
    repo, paper, res = _mini_repo(tmp_path, designs['A'])
    registry, reviews = repo / 'autopilot' / 'release_assets.json', paper / 'numeric_reviews.json'
    auto = paper / 'auto'
    kw = dict(registry=registry, review_file=reviews, repo_root=repo)
    pf.rehearsal_bind(auto / 'cr_results.values.json', reviews, repo)  # as build --update-reviews
    with pytest.raises(mk.InputError, match='integrate'):  # no figure float yet
        mk.register_figure(paper, auto, 'Reviewer', **kw)
    mk.integrate(paper, auto)
    unbound = tmp_path / 'unbound.json'
    shutil.copyfile(mk.PAPER / 'numeric_reviews.json', unbound)
    with pytest.raises(mk.InputError, match='not bound'):
        mk.register_figure(paper, auto, 'Reviewer', registry=registry, review_file=unbound,
                           repo_root=repo)
    with pytest.raises(mk.InputError, match='reviewer'):
        mk.register_figure(paper, auto, '  ', **kw)
    real = {p: mk.sha256_file(p) for p in (mk.REGISTRY, mk.PAPER / 'numeric_reviews.json')}
    for bad in (dict(kw, registry=mk.REGISTRY), dict(kw, review_file=mk.PAPER / 'numeric_reviews.json')):
        with pytest.raises(mk.InputError, match='refusing to register'):
            mk.register_figure(paper, auto, 'Reviewer', allow_real_paper=True, **bad)
    assert real == {p: mk.sha256_file(p) for p in real}
    old_text = registry.read_bytes()
    done = mk.register_figure(paper, auto, 'Synthetic test reviewer', **kw)
    rel = 'auto/fig_cr_seed_spread.png'
    assert done['graphic'] == rel
    new_text = registry.read_bytes()
    old, new = json.loads(old_text), json.loads(new_text)
    entry = new['producers'].pop(rel)
    assert new == old  # nothing else changed
    head = old_text.index(b'"producers"')
    assert new_text[:head] == old_text[:head] and b'\r\n' in new_text
    assert b'\n' not in new_text.replace(b'\r\n', b'')
    png = auto / 'fig_cr_seed_spread.png'
    evidence = json.load(open(auto / 'fig_cr_seed_spread.evidence.json'))
    assert entry['expected_output_sha256'] == mk.sha256_file(png) == evidence['outputs']['png']
    assert entry['companions'] == {'auto/fig_cr_seed_spread.pdf': evidence['outputs']['pdf']}
    assert entry['evidence_manifest'] == 'paper/genai4health2026/auto/fig_cr_seed_spread.evidence.json'
    assert entry['synthetic'] is True and entry['script'] == 'autopilot/make_cr_results.py'
    for path in entry['inputs']:
        assert (repo / path).is_file()
    # The producer pin works as a release_assets declaration ...
    tiny = tmp_path / 'tiny'
    (tiny / 'auto').mkdir(parents=True)
    (tiny / 'main_submission.tex').write_text(r'\includegraphics{fig_cr_seed_spread.png}')
    shutil.copyfile(png, tiny / 'auto' / 'fig_cr_seed_spread.png')
    inventory = assets.asset_inventory(tiny, registry)
    assert inventory['ALL_PASS'] and inventory['items'][0]['identity_status'] == 'generated_input_hashed'
    (tiny / 'auto' / 'fig_cr_seed_spread.png').write_bytes(b'changed')
    assert not assets.asset_inventory(tiny, registry)['ALL_PASS']
    # ... and the receipt satisfies p15's figure-receipt check against the integrated caption.
    ev = numeric.Evidence(paper, tmp_path, numeric.BUILTIN_SOURCES)
    ev.roots['repo'] = repo
    review_data = numeric.read_reviews(paper, ev, reviews)
    receipts = [f for f in review_data['figures'] if f['path'] == rel]
    assert len(receipts) == 1 and receipts[0]['validation']['reviewer'] == 'Synthetic test reviewer'
    text, _ = assets.source_tree(paper)
    caption = numeric.figure_context(text, rel)
    assert evidence['points'] and r'\caption{' not in caption
    item = {'path': rel, 'sha256': mk.sha256_file(png)}
    assert numeric.figure_receipt(item, receipts[0], ev, caption)['status'] == \
        'reviewed_source_illustration'
    # Re-registration is idempotent (one producer, one receipt, identical registry bytes).
    mk.register_figure(paper, auto, 'Synthetic test reviewer', **kw)
    assert registry.read_bytes() == new_text
    assert len([f for f in json.load(open(reviews, encoding='utf-8'))['figures']
                if f['path'] == rel]) == 1


def test_i12_tldr_exists_for_every_scenario_and_never_exceeds_its_label(designs, tmp_path):
    for key in ('A', 'B', 'C', 'I', 'mixed', 'na', 'noC1_late'):
        res = run_build(designs[key], tmp_path / key)
        tldr = (tmp_path / key / 'out' / 'cr_tldr.txt').read_text(encoding='utf-8').strip()
        assert tldr and tldr == res['tldr'], key
        if key == 'A':  # two complete indices, none incomplete
            assert 'in every repeated continuation' in tldr
    d = designs['mixed']
    ctx, _, _ = mk.prepare(d['final'], d['runs'], d['md'], [I5],
                           evidence_dir=os.path.join(d['root'], '_ev_tldr'), allow_synthetic=True)
    for c in mk.LABEL_CODE:
        for e in mk.LABEL_CODE:
            for m in ('M+', 'M0', 'M-', 'M_partial', 'M_desc', 'M_na', 'M_unassessed', 'M_noP'):
                ctx.labels = {'centroid': c, 'envelope': e}
                ctx.scenario, ctx.codes = mk.scenario_of(ctx.labels)
                ctx.matched = dict(ctx.matched, label=m)
                tldr = mk.tldr_text(ctx)
                codes = set(ctx.codes.values())
                assert tldr and len(tldr) <= mk.TLDR_LIMIT and tldr.isascii(), (c, e, m)
                assert not re.search(r'\d', tldr), (c, e, m)
                if 'A' not in codes:  # no gain claim without a consistent-direction label
                    assert 'in every repeated continuation' not in tldr and \
                        'keeps its gain' not in tldr, (c, e, m)
                if ctx.block['incomplete_seed_indices']:  # label covers the complete indices
                    assert 'in every repeated continuation' not in tldr, (c, e, m)
                if ctx.scenario == 'I':  # incomplete: only the original single runs
                    assert 'in single runs' in tldr and 'repeated' not in tldr, (c, e, m)
                if ctx.scenario == 'mixed' and 'I' in codes:
                    assert 'is incomplete' in tldr, (c, e, m)
                # V8-4: the labels compare each guide with RANDOM, never the guides with each
                # other, and say nothing about anatomical detail.
                assert not [b for b in GUIDE_COMPARISONS if b in tldr.lower()], (c, e, m)
                text = mk.build_text(ctx)
                prose = ' '.join(text.macros.values()).lower()
                assert not [b for b in GUIDE_COMPARISONS if b in prose], (c, e, m)
                if ctx.block['incomplete_seed_indices']:
                    for name in ('RepAbstractResults', 'RepIntroResults'):
                        assert not re.search(r'(?<!not )in every continuation',
                                             text.macros[name]), (c, e, m, name)
                if m in ('M+', 'M_partial', 'M-'):  # one continuation per arm (amendment 2)
                    for name in ('RepAbstractResults', 'RepIntroResults'):
                        assert 'single continuation' in text.macros[name], (c, e, m, name)


GUIDE_COMPARISONS = ('anatomical detail', 'help further', 'better than the intensity',
                     'better than segmentation', 'outperforms', 'than envelope', 'than centroid',
                     'more anatomically specific strategies do not')


def test_v8_tldr_makes_no_guide_comparison_when_envelope_gains_more(tmp_path, monkeypatch):
    """V8-4 producer-level counterexample: ENVELOPE gains exceed CENTROID's in every complete
    index, both labels consistent direction (real cr_stats unseal/final, no label edits)."""
    monkeypatch.setitem(fx.PATTERNS, 'Aenv', {'centroid': (0.0074, 0.0077, 0.0069),
                                              'envelope': (0.0190, 0.0200, 0.0190)})
    d = fx.build_design(tmp_path / 'd', 'A', 'Aenv', 'M+', seeds=(1234, 5678, 9012),
                        n_test=N_TEST, n_boot=N_BOOT)
    final = json.load(open(d['final']))
    out = final['analysis'][fx.PRIMARY]['outcomes']
    assert out['centroid']['label'] == out['envelope']['label'] == 'consistent_direction'
    assert all(e > c for c, e in zip(out['centroid']['new_seed_deltas'],
                                     out['envelope']['new_seed_deltas']))
    res = run_build(d, tmp_path / 'b')
    assert res['scenario']['scenario'] == 'A'
    tldr = res['tldr']
    assert not [b for b in GUIDE_COMPARISONS if b in tldr.lower()], tldr
    assert tldr.endswith('than uniform masking in every repeated continuation.')
    texts = {t['name']: t['text'] for t in res['text_macros']}
    assert 'In the original runs, more anatomically specific strategies' in \
        texts['RepAbstractResults']
    assert 'mixed single-run results for more specific anatomical targeting' in \
        texts['RepContribResults']


def test_v8_stale_final_is_refused_when_completion_evidence_changes(tmp_path):
    """V8-2: final.json computed with an on-time E3 stop file must be refused once that stop
    file is missing or changed (the recomputed eligibility differs)."""
    d = fx.build_design(tmp_path / 'd', 'A', 'A', 'M+', seeds=(1234, 5678, 9012),
                        n_test=N_TEST, n_boot=N_BOOT)
    run_build(d, tmp_path / 'ok')
    stop = os.path.join(d['root'], 'probes', 'train_envelope_s9012', 'stop_epoch_050.json')
    good = open(stop, 'rb').read()
    os.remove(stop)
    with pytest.raises(mk.InputError, match='eligibility is stale'):
        run_build(d, tmp_path / 'missing')
    with open(stop, 'wb') as stream:
        stream.write(good.replace(b'2026-10-20T12:00:00-07:00', b'2026-10-22T09:30:00-07:00'))
    with pytest.raises(mk.InputError, match='eligibility is stale'):
        run_build(d, tmp_path / 'changed')
    with open(stop, 'wb') as stream:
        stream.write(good)
    run_build(d, tmp_path / 'restored')
    # MD JSON carrying another eligibility than final is refused too.
    _edit_json(d['md'], lambda m: next(iter(m['runs'].values()))['eligibility'].update(
        eligible=False))
    with pytest.raises(mk.InputError, match='eligibility differs from the final analysis'):
        run_build(d, tmp_path / 'md')


def _edited(d, tmp_path, name, edit):
    """Design d with an edited copy of its final JSON (the module fixture stays untouched)."""
    final = json.load(open(d['final']))
    edit(final)
    path = tmp_path / ('%s_final.json' % name)
    path.write_text(json.dumps(final))
    return dict(d, final=str(path))


def _flip(outcome):
    outcome['label'] = ('consistent_direction' if outcome['label'] != 'consistent_direction'
                        else 'reversed')


def test_v10_edited_final_labels_or_values_are_refused(designs, tmp_path):
    """V10-1: labels, indices, contrasts, means and ranges are recomputed from the verified
    per-run AUCs with the registered rules; an edited final JSON is refused (build and the
    --check CLI), never turned into prose."""
    def pri(f):
        return f['analysis'][fx.PRIMARY]

    def flag(f):
        c = pri(f)['matched_control'][0]
        c['abs_delta_exceeds_random_range'] = not c['abs_delta_exceeds_random_range']

    edits = {
        'labels_C_to_A': ('C', lambda f: [o.update(label='consistent_direction')
                                          for o in pri(f)['outcomes'].values()]),
        'label_B_to_A_mixed': ('mixed', lambda f: pri(f)['outcomes']['envelope'].update(
            label='consistent_direction')),
        'secondary_label': ('A', lambda f: _flip(
            f['analysis']['patchmean_slicemax']['outcomes']['envelope'])),
        'mean': ('B', lambda f: pri(f)['outcomes']['centroid'].update(mean_delta=0.02)),
        'delta': ('B', lambda f: pri(f)['primary_contrasts'][0].update(delta=0.02)),
        'range': ('B', lambda f: [o.update(random_range=0.0001)
                                  for o in pri(f)['outcomes'].values()]),
        'indices': ('mixed', lambda f: pri(f).update(complete_seed_indices=[1234, 5678, 9012],
                                                     incomplete_seed_indices=[])),
        'dropped_contrast': ('A', lambda f: pri(f)['primary_contrasts'].pop()),
        'matched_flag': ('A', flag),
        'val_auc': ('A', lambda f: pri(f)['per_run_val_auc'].update(
            {sorted(pri(f)['per_run_val_auc'])[0]: 0.5})),
    }
    for name, (key, edit) in edits.items():
        d = _edited(designs[key], tmp_path, name, edit)
        with pytest.raises(mk.InputError, match='recomputed from the verified per-run AUCs'):
            run_build(d, tmp_path / name)
    # V10's counterexample through the real CLI: C/C relabelled A/A, every number unchanged.
    d = _edited(designs['C'], tmp_path, 'cli', edits['labels_C_to_A'][1])
    assert mk.main(['build', '--final', d['final'], '--runs'] + d['runs'] +
                   ['--md', d['md'], '--budget-i5', I5, '--out-dir', str(tmp_path / 'cli' / 'o'),
                    '--evidence-dir', str(tmp_path / 'cli' / 'e'), '--allow-synthetic',
                    '--no-figure', '--check']) == 2
    run_build(designs['C'], tmp_path / 'untouched')  # the unedited final still builds


def _texts(res):
    return {t['name']: t['text'] for t in res['text_macros']}


def test_v10_missing_or_late_original_random_withholds_registered_labels(tmp_path, monkeypatch):
    """V10-2 / V10-3, producer level (real unseal / final): all ten new runs on time, three
    complete indices; original RANDOM present, missing or late. Without it there is no
    registered RANDOM range: the matched control stays descriptive (M_incomplete) although the
    range of the remaining RANDOM runs would have turned M0 into M+, and every text names the
    missing reference instead of unfinished continuations."""
    monkeypatch.setitem(fx.RANDOM, None, 0.82)
    kw = dict(seeds=(1234, 5678, 9012), n_test=N_TEST, n_boot=N_BOOT, cb_placement=0.00555)
    res = run_build(fx.build_design(tmp_path / 'with', 'A', 'A', 'M0', **kw), tmp_path / 'b_with')
    within = 'within_run_to_run_variation'
    assert res['scenario']['labels'] == {'centroid': within, 'envelope': within}
    assert res['scenario']['matched_control']['label'] == 'M0'
    assert res['scenario']['reference_missing'] == []
    for case, extra, why, state in (
            ('missing', dict(drop=[('random', None)]), 'anchor_missing', 'is not available'),
            ('late', dict(late=[('random', None)]), 'anchor_late',
             'finished after the results freeze')):
        d = fx.build_design(tmp_path / case, 'A', 'A', 'M0', **dict(kw, **extra))
        block = json.load(open(d['final']))['analysis'][fx.PRIMARY]
        assert block['complete_seed_indices'] == [1234, 5678, 9012]
        assert {o['label'] for o in block['outcomes'].values()} == {'incomplete'}
        partial = block['outcomes']['centroid']['random_range']  # remaining RANDOM runs only
        out = tmp_path / ('b_' + case)
        res = run_build(d, out)
        mc = res['scenario']['matched_control']
        assert partial is not None and mc['P'] > partial and \
            mc['P'] >= 0.5 * (mc['P'] + mc['B'])  # would have read as M+
        assert mc['label'] == 'M_incomplete' and mc['name'] is None and mc['spread'] is None
        assert mc['reference_missing'] == [why] == res['scenario']['reference_missing']
        assert res['scenario']['scenario'] == 'I'
        tex = (out / 'out' / 'cr_results.tex').read_text(encoding='utf-8')
        assert '\\newcommand{\\RepRandomRange}' not in tex
        texts = _texts(res)
        assert 'All ten registered continuations finished before the results freeze.' in \
            texts['RepSetupResults']
        assert ('original \\textsc{random} encoder, the reference of the registered rule, '
                + state) in texts['RepSetupResults']
        for name in ('RepAbstractResults', 'RepIntroResults', 'RepResultsSeeds',
                     'RepDiscussionLead', 'RepLimitations', 'RepConclusion'):
            assert 'did not all finish' not in texts[name] and \
                'did not finish' not in texts[name], (case, name)
            assert 're-probed original' in texts[name] and state in texts[name], (case, name)
        control = texts['RepResultsControl']
        assert 'No registered description is assigned because the re-probed original' in control
        assert 'Its registered description' not in control
        # mixed branches with an I arm state the same actual reason
        ctx, _, _ = mk.prepare(d['final'], d['runs'], d['md'], [I5],
                               evidence_dir=os.path.join(d['root'], '_ev_mixed'),
                               allow_synthetic=True)
        ctx.labels = {'centroid': 'reversed', 'envelope': 'incomplete'}
        ctx.scenario, ctx.codes = mk.scenario_of(ctx.labels)
        macros = mk.build_text(ctx).macros
        assert ctx.scenario == 'mixed'
        assert ('could not be assessed because the re-probed original uniform-masking encoder'
                in macros['RepAbstractResults'])
        assert 'has no registered label because the re-probed original' in \
            macros['RepIntroResults']
        assert all('did not all finish' not in macros[n] for n in
                   ('RepAbstractResults', 'RepIntroResults', 'RepLimitations'))


def _literal(path, name):
    """A module-level dict literal of another script, read without importing it."""
    import ast
    for node in ast.parse(open(path, encoding='utf-8').read()).body:
        if isinstance(node, ast.Assign) and any(getattr(x, 'id', None) == name
                                                for x in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _pdf_font_sizes(path):
    import zlib
    raw = open(path, 'rb').read()
    sizes = []
    for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', raw, re.S):
        try:
            body = zlib.decompress(match.group(1))
        except zlib.error:
            body = match.group(1)
        sizes += [float(s) for s in re.findall(rb'/F\d+ ([\d.]+) Tf', body)]
    return sizes


POINT_KEYS = {'run', 'arm', 'tag', 'seed', 'x', 'test_auc', 'head_sd', 'status'}


def test_f3_figure_uses_the_paper_palette_and_records_its_contract(designs, tmp_path):
    """F3: arm colours / markers are p8's single mapping (RANDOM-CB uses neither a colour nor a
    marker of any p8 arm), seeds are letters at x offsets, text prints at >= 7 pt at the
    included width, panel (b) holds the final's paired differences with the registered band,
    and the plotted data (points, means) keep the schema the poster gate checks."""
    p8 = os.path.join(REPO, 'autopilot', 'p8_make_assets.py')
    col, mk_ = _literal(p8, 'COL'), _literal(p8, 'MK')
    story = _literal(os.path.join(REPO, 'paper', 'genai4health2026', 'scripts',
                                  'make_story_figures.py'), 'COLORS')
    assert {a: mk.FIG_COLORS[a] for a in ('random', 'centroid', 'envelope')} == \
        {'random': col['random'], 'centroid': col['oracle'], 'envelope': col['envelope']}
    assert mk.FIG_COLORS['random_cb'].lower() not in {c.lower() for c in
                                                      list(col.values()) + list(story.values())}
    assert {a: mk.FIG_ARM_MARKERS[a] for a in ('random', 'centroid', 'envelope')} == \
        {'random': mk_['random'], 'centroid': mk_['oracle'], 'envelope': mk_['envelope']}
    assert mk.FIG_ARM_MARKERS['random_cb'] not in mk_.values()
    assert set(mk.FIG_SEED_LETTERS.values()) == {'O', 'A', 'B', 'C'}
    d = designs['mixed']  # E3 late: seed C descriptive; registered range exists
    run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    ev = json.load(open(out / 'fig_cr_seed_spread.evidence.json'))
    assert ev['encoding'] == json.loads(json.dumps(mk.figure_encoding()))
    assert all(set(p) == POINT_KEYS for p in ev['points'])
    assert all(set(m) == {'x', 'mean', 'n'} for m in ev['means'].values())
    block = json.load(open(d['final']))['analysis'][fx.PRIMARY]
    panel = ev['panel_b']
    assert panel['band'] == {'drawn': True, 'r': block['outcomes']['centroid']['random_range'],
                             'missing': []}
    for q in panel['points']:
        listname, i = q['contrast']
        c = block[listname][i]
        assert q['delta'] == c['delta'] and c['kind'] == q['arm'] + '-random'
        assert q['status'] == ('registered' if listname == 'primary_contrasts' else 'descriptive')
        assert (q['status'] == 'descriptive') == (q['seed'] == 9012)
    assert {a: m['mean_delta'] for a, m in panel['means'].items()} == \
        {a: block['outcomes'][a]['mean_delta'] for a in ('centroid', 'envelope')}
    assert len(panel['points']) == 6
    caption = tex_macros(out / 'cr_results.tex')['RepFigSeedSpreadCaption']
    assert 'grey band' in caption and r'\RepRandomRange' in caption
    assert '(a)~' in caption and '(b)~' in caption and not re.search(r'\d', caption)
    tex = (out / 'cr_figure_seed_spread.tex').read_text(encoding='utf-8')
    assert r'\includegraphics[width=\linewidth]{fig_cr_seed_spread.png}' in tex
    # printed 1:1 (figure width = 5.5 in text width): every font size is >= 7 pt
    assert mk.FIG_SIZE_IN[0] == ev['encoding']['include_width_in'] == 5.5
    sizes = _pdf_font_sizes(out / 'fig_cr_seed_spread.pdf')
    assert sizes and min(sizes) >= 7.0, sorted(set(sizes))
    # --check recomputes panel (b) and the contract
    for edit in (lambda e: e['panel_b']['points'][0].update(delta=0.5),
                 lambda e: e['panel_b']['band'].update(drawn=False),
                 lambda e: e['encoding']['colors'].update(centroid='#0072B2')):
        good = (out / 'fig_cr_seed_spread.evidence.json').read_bytes()
        _edit_json(out / 'fig_cr_seed_spread.evidence.json', edit)
        assert 'fig_cr_seed_spread.evidence.json' in run_build(d, tmp_path, check=True,
                                                               figure=True)['stale']
        (out / 'fig_cr_seed_spread.evidence.json').write_bytes(good)
    assert run_build(d, tmp_path, check=True, figure=True)['stale'] == []


def test_f3_no_band_without_the_registered_range(designs, tmp_path):
    d = designs['I']  # one complete index: no registered RANDOM range
    run_build(d, tmp_path, figure=True)
    out = tmp_path / 'out'
    ev = json.load(open(out / 'fig_cr_seed_spread.evidence.json'))
    assert ev['panel_b']['band'] == {'drawn': False, 'r': None, 'missing': ['indices']}
    caption = tex_macros(out / 'cr_results.tex')['RepFigSeedSpreadCaption']
    assert 'no band is drawn because the registered range' in caption
    assert r'\pm' not in caption and r'\RepRandomRange' not in caption


def test_h8_md_secondary_analysis_is_reported_or_declared_not_run(designs, tmp_path):
    d = designs['A']
    res = run_build(d, tmp_path / 'with')
    texts = {t['name']: t['text'] for t in res['text_macros']}
    assert res['md_secondary'] == {'run': True, 'source': 'rep_md'}
    assert r'Table~\ref{tab:md}' in texts['RepAppendixResults']
    assert r'\label{tab:md}' in (tmp_path / 'with' / 'out' / 'cr_table_md.tex').read_text()
    res = mk.build(d['final'], d['runs'], None, [I5], out_dir=tmp_path / 'without' / 'out',
                   evidence_dir=tmp_path / 'without' / 'ev', allow_synthetic=True, figure=False)
    texts = {t['name']: t['text'] for t in res['text_macros']}
    assert res['md_secondary'] == {'run': False, 'source': None}
    assert 'mean-deviation regression within glaucoma cases, a secondary analysis, was not run' \
        in texts['RepAppendixResults']
    assert any('MD secondary analysis not supplied' in w for w in res['warnings'])


def test_i12_generated_table_cells_keep_their_words_for_the_word_gate(designs, tmp_path):
    import check_docx
    run_build(designs['A'], tmp_path, figure=False)
    for path in sorted((tmp_path / 'out').glob('cr_table_*.tex')):
        body = assets.uncomment(path.read_text(encoding='utf-8'))
        # check_docx reads "$\Delta$MAE" as one control word; math must not touch a word.
        assert not re.search(r'(?<!\\)\$[^$]+\$[A-Za-z]', body), path.name
    md = (tmp_path / 'out' / 'cr_table_md.tex').read_text(encoding='utf-8')
    header = [line for line in md.splitlines() if line.startswith('policy &')][0]
    words = [check_docx.cell_words(cell) for cell in header.rstrip('\\ ').split('&')]
    assert words[5] == ('mae', 'interval') and words[6] == ('spearman',)
