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
                'C': ('C', 'C', 'C', 'M-'), 'I': ('I', 'I', 'I', 'M+'),
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
    outcomes = {arm: {'random_range': spread} for arm in ('centroid', 'envelope')}
    if descriptive:
        contrasts, late = [], contrasts + late
    return {'outcomes': outcomes, 'primary_contrasts': [], 'matched_control': contrasts,
            'descriptive_incomplete_contrasts': late}


def test_matched_control_rule_and_boundaries():
    """PREREGISTRATION amendment 2, labels only from on-time runs at seed 1234."""
    cb_run = {'cb': {'arm': 'random_cb', 'seed': 1234, 'eligible': True}}
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
    assert open(out / 'cr_figure_seed_spread.tex').read().count(r'\RepFigSeedSpreadCaption') == 1


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
