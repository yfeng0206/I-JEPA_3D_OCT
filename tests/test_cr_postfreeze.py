"""autopilot/cr_postfreeze.py, cr_postfreeze.ps1 guards and cr_migrate_receipts.py (I12).

Synthetic campaigns from tests/cr_postfreeze_fixtures.py (sealed probe dirs, state.json with the
runner's hashes, stop files). CPU only; no real campaign, run or paper file is read or written
except the read-only paper/registry copies made by the helpers under test.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'autopilot'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault('MPLBACKEND', 'Agg')
import cr_migrate_receipts as mig  # noqa: E402
import cr_postfreeze as pf  # noqa: E402
import cr_postfreeze_fixtures as pfx  # noqa: E402
import cr_stats  # noqa: E402
import numeric_bindings as numeric  # noqa: E402
import release_assets as assets  # noqa: E402

SMALL = dict(n_test=120, n_train=120, n_val=60)


@pytest.fixture(scope='module')
def campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp('pfc')
    return {
        'ontime': pfx.build_campaign(root / 'ontime', **SMALL),
        'late': pfx.build_campaign(root / 'late', late=[('envelope', 9012)], **SMALL),
        'inprog': pfx.build_campaign(root / 'inprog', in_progress=[('envelope', 9012)], **SMALL),
        'nocb': pfx.build_campaign(root / 'nocb', centroid='B', envelope='B', matched='M_na',
                                   **SMALL),
    }


def plan_of(c, **kw):
    return pf.make_plan(c['campaign'], c['state'], rehearsal=True, fixture_root=c['root'], **kw)


def by_id(plan):
    return {r['id']: r for r in plan['runs']}


def test_plan_classifies_ready_late_in_progress_and_absent_runs(campaigns, tmp_path):
    plan = plan_of(campaigns['ontime'])
    assert len(plan['ready_dirs']) == 13 and len(plan['to_unseal']) == 13
    assert plan['late'] == [] and plan['in_progress'] == [] and plan['blocking'] == []
    assert all(info['primary_on_time'] for info in plan['seed_indices'].values())
    assert plan['cb']['status'] == 'ready' and len(plan['cb']['train_logs']) == 1

    plan = plan_of(campaigns['late'])
    runs = by_id(plan)
    assert plan['late'] == ['E3'] and runs['E3']['status'] == 'ready'
    assert any('after the freeze' in r for r in runs['E3']['eligibility']['reasons'])
    assert not plan['seed_indices']['9012']['primary_on_time']
    assert plan['seed_indices']['1234']['primary_on_time']

    c = campaigns['inprog']
    plan = plan_of(c)
    runs = by_id(plan)
    assert plan['in_progress'] == ['E3'] and runs['E3']['status'] == 'in_progress'
    assert len(plan['ready_dirs']) == 12
    assert [s['reason'] for s in plan['stray_dirs']] == ['in-progress attempt of E3']
    args = dict(campaign=c['campaign'], state=c['state'], runs_root=None,
                out=str(tmp_path / 'plan.json'), rehearsal=True, fixture_root=c['root'],
                inputs=[])
    assert pf.cmd_plan(argparse.Namespace(allow_in_progress=False, **args)) == 3
    assert pf.cmd_plan(argparse.Namespace(allow_in_progress=True, **args)) == 0

    plan = plan_of(campaigns['nocb'])
    runs = by_id(plan)
    assert runs['CB']['status'] == 'not_completed' and plan['not_completed'] == ['CB']
    assert plan['cb']['status'] == 'not_completed' and len(plan['ready_dirs']) == 12


def test_plan_late_is_exactly_cr_stats_final_eligibility(tmp_path):
    c = pfx.build_campaign(tmp_path / 'c', late=[('envelope', 9012), ('random_cb', 1234)],
                           matched='M_desc', **SMALL)
    plan = plan_of(c)
    key0 = pf.input_key(_write(tmp_path / 'p0.json', plan), [])
    for run_dir in plan['to_unseal']:
        cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    final = cr_stats.cmd_final(plan['ready_dirs'], n_boot=50, allow_before_freeze=True)
    planned = {os.path.basename(r['probe_dir']): not r['late'] for r in plan['runs']
               if r['status'] == 'ready'}
    assert planned == {name: run['eligibility']['eligible'] for name, run in final['runs'].items()}
    assert sorted(by_id(plan)[i]['late'] for i in ('E3', 'CB')) == [True, True]
    block = final['analysis'][cr_stats.PRIMARY_VARIANT]
    assert block['incomplete_seed_indices'] == [9012]
    # After unsealing, the plan reports the receipts (never unsealed twice by the pipeline),
    # and the analysis cache key changes (final / MD are recomputed), then stays stable.
    again = plan_of(c)
    assert again['to_unseal'] == [] and len(again['already_unsealed']) == 13
    key = pf.input_key(_write(tmp_path / 'p1.json', again), [])
    assert key != key0
    assert key == pf.input_key(_write(tmp_path / 'p2.json', plan_of(c)), [])
    assert key != pf.input_key(_write(tmp_path / 'p3.json', again), ['n_boot=1'])


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return str(path)


def test_plan_refuses_tampered_inputs_and_mode_mismatch(tmp_path):
    c = pfx.build_campaign(tmp_path / 'c', seeds=(1234, 5678), **SMALL)
    with pytest.raises(pf.Refused, match='without rehearsal'):
        pf.make_plan(c['campaign'], c['state'], rehearsal=False)
    # Real-like campaign/state (no synthetic flag) inside the fixture root: refused in rehearsal.
    real_like = os.path.join(c['root'], 'campaign_real.json')
    data = json.load(open(c['campaign']))
    data.pop('synthetic')
    with open(real_like, 'w') as stream:
        json.dump(data, stream)
    state = json.load(open(c['state']))
    state.pop('synthetic')
    real_state = os.path.join(c['root'], 'state_real.json')
    with open(real_state, 'w') as stream:
        json.dump(state, stream)
    with pytest.raises(pf.Refused, match='needs campaign.json and state.json'):
        pf.make_plan(real_like, real_state, rehearsal=True, fixture_root=c['root'])
    with pytest.raises(pf.Refused, match='needs campaign.json and state.json'):
        pf.make_plan(c['campaign'], real_state, rehearsal=True, fixture_root=c['root'])
    r1 = by_id(plan_of(c))['R1']['probe_dir']
    with open(os.path.join(r1, 'results.json'), 'a') as stream:
        stream.write(' ')
    plan = plan_of(c)
    assert by_id(plan)['R1']['status'] == 'invalid'
    assert any('changed since the campaign validated it' in b for b in plan['blocking'])
    args = argparse.Namespace(campaign=c['campaign'], state=c['state'], runs_root=None,
                              out=str(tmp_path / 'plan.json'), rehearsal=True,
                              allow_in_progress=True, fixture_root=c['root'], inputs=[])
    assert pf.cmd_plan(args) == 2


def test_rehearsal_helpers_refuse_the_real_repository(tmp_path):
    with pytest.raises(pf.Refused, match='outside the repository'):
        pf.scratch_copy(assets.PAPER, assets.REPO / '_i12_never', 'paper')
    with pytest.raises(pf.Refused, match='never write inside the real repository'):
        pf.rehearsal_bind(tmp_path / 'x.json', assets.PAPER / 'numeric_reviews.json', tmp_path)
    with pytest.raises(pf.Refused, match='never write inside the real repository'):
        pf.rehearsal_ack(assets.PAPER)
    with pytest.raises(pf.Refused, match='outside the repository'):
        pf.rehearsal_mirror(assets.REPO, assets.REPO / '_i12_mirror')
    with pytest.raises(pf.Refused, match='not inside a rehearsal mirror'):
        pf.rehearsal_ack(tmp_path)
    (tmp_path / 'dst').mkdir()
    with pytest.raises(pf.Refused, match='not a scratch copy'):
        pf.scratch_copy(tmp_path / 'src', tmp_path / 'dst', 'paper')
    assert not (assets.REPO / '_i12_never').exists() and not (assets.REPO / '_i12_mirror').exists()


@pytest.mark.skipif(shutil.which('powershell') is None, reason='Windows PowerShell not available')
def test_ps1_refuses_real_paths_in_rehearsal_and_any_run_before_the_freeze(tmp_path):
    script = os.path.join(REPO, 'autopilot', 'cr_postfreeze.ps1')

    def run(*args):
        return subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                               script] + list(args), capture_output=True, text=True, timeout=300)

    out = tmp_path / 'out'
    for fixture in (r'D:\jepa_phase0\runs\cr_i12_never', r'D:\jepa_phase0\campaign\cr_i12_never',
                    os.path.join(REPO, 'tests')):
        proc = run('-Rehearsal', '-FixtureRoot', fixture, '-OutRoot', str(out),
                   '-TargetRepo', str(tmp_path / 'm'))
        assert proc.returncode != 0 and 'Rehearsal refuses real path' in proc.stdout + proc.stderr
    proc = run('-Rehearsal', '-FixtureRoot', str(tmp_path / 'fx'), '-OutRoot', str(out),
               '-TargetRepo', REPO)
    assert proc.returncode != 0 and 'refuses real path' in proc.stdout + proc.stderr
    if cr_stats._now() < __import__('datetime').datetime.fromisoformat(cr_stats.FREEZE):
        proc = run('-OutRoot', str(out))
        assert proc.returncode != 0 and 'before the results freeze' in proc.stdout + proc.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# V8-1: rehearsal containment of every derived path
# ---------------------------------------------------------------------------

def _redirect_state(c, run_id, out_dir):
    """Point one run of a fixture state at another probe dir (hashes recorded as the runner
    would, so only containment can stop it)."""
    state = json.load(open(c['state']))
    result = state['runs'][run_id]['probe']['result']
    result['out_dir'] = str(out_dir)
    for key, name in (('aggregate', 'results.json'), ('sealed_manifest', 'sealed_manifest.json')):
        result['files'][key] = {'path': os.path.join(str(out_dir), name),
                                'sha256': assets.sha256(os.path.join(str(out_dir), name))}
    with open(c['state'], 'w') as stream:
        json.dump(state, stream)


def _c1_dir(c):
    return by_id(pf.make_plan(c['campaign'], c['state'], rehearsal=True,
                              fixture_root=c['root']))['C1']['probe_dir']


@pytest.fixture()
def surrogate(tmp_path):
    """Fixture A whose C1 entry points at C1 of a separate synthetic campaign F outside A."""
    a = pfx.build_campaign(tmp_path / 'A', seeds=(1234, 5678), **SMALL)
    f = pfx.build_campaign(tmp_path / 'F', seeds=(1234, 5678), **SMALL)
    foreign = _c1_dir(f)
    _redirect_state(a, 'C1', foreign)
    return a, f, foreign


def test_rehearsal_plan_refuses_probe_paths_outside_the_fixture_root(surrogate, tmp_path):
    a, _, foreign = surrogate
    with pytest.raises(pf.Refused, match='outside the fixture root'):
        pf.make_plan(a['campaign'], a['state'], rehearsal=True, fixture_root=a['root'])
    assert not os.path.exists(os.path.join(foreign, 'unsealed_results.json'))
    # A junction inside the fixture's run root that resolves to the foreign dir is refused too.
    import _winapi
    link = os.path.join(a['runs_root'], 'cr_probe_C1_link_ep050_a0')
    _winapi.CreateJunction(foreign, link)
    try:
        assert os.path.isfile(os.path.join(link, 'results.json'))  # the junction resolves
        _redirect_state(a, 'C1', link)
        with pytest.raises(pf.Refused, match='outside the fixture root'):
            pf.make_plan(a['campaign'], a['state'], rehearsal=True, fixture_root=a['root'])
        with pytest.raises(pf.Refused, match='outside the fixture root'):
            pf.contain(a['root'], [link])
    finally:
        if os.path.isdir(link):
            os.rmdir(link)  # removes the junction only
    assert os.path.isfile(os.path.join(foreign, 'results.json'))


def test_rehearsal_plan_refuses_escaping_provenance_markers_and_real_roots(tmp_path):
    c = pfx.build_campaign(tmp_path / 'A', seeds=(1234, 5678), **SMALL)
    f = pfx.build_campaign(tmp_path / 'F', seeds=(1234, 5678), **SMALL)
    # stop file of R1 outside the fixture (results.json edited and re-hashed in the state)
    r1 = by_id(pf.make_plan(c['campaign'], c['state'], rehearsal=True,
                            fixture_root=c['root']))['R1']['probe_dir']
    res_path = os.path.join(r1, 'results.json')
    good = open(res_path, 'rb').read()
    res = json.loads(good)
    res['identity']['run_provenance'] = os.path.join(f['runs_root'], 'cr_seed_v1_random_s1234',
                                                     'stop_epoch_050.json')
    with open(res_path, 'w') as stream:
        json.dump(res, stream)
    _redirect_state(c, 'R1', r1)
    with pytest.raises(pf.Refused, match='run provenance'):
        pf.make_plan(c['campaign'], c['state'], rehearsal=True, fixture_root=c['root'])
    with open(res_path, 'wb') as stream:
        stream.write(good)
    _redirect_state(c, 'R1', r1)
    pf.make_plan(c['campaign'], c['state'], rehearsal=True, fixture_root=c['root'])
    # explicit boolean synthetic provenance
    for path, edit in ((c['campaign'], lambda d: d.update(synthetic='true')),
                       (os.path.join(c['root'], pfx.FIXTURE_MARKER), lambda d: d.pop('synthetic'))):
        raw = open(path, 'rb').read()
        data = json.loads(raw)
        edit(data)
        with open(path, 'w') as stream:
            json.dump(data, stream)
        with pytest.raises(pf.Refused, match='synthetic'):
            pf.make_plan(c['campaign'], c['state'], rehearsal=True, fixture_root=c['root'])
        with open(path, 'wb') as stream:
            stream.write(raw)
    # real roots, refused by string before any path is resolved or read
    for root in (r'D:\jepa_phase0\runs\cr_i12_never', r'D:\jepa_phase0\campaign\x',
                 r'D:\jepa_phase0\cr_worktrees\x', r'D:\jepa_phase0', str(assets.REPO)):
        with pytest.raises(pf.Refused, match='real root'):
            pf.make_plan(c['campaign'], c['state'], rehearsal=True, fixture_root=root)
    with pytest.raises(pf.Refused, match='fixture-root'):
        pf.make_plan(c['campaign'], c['state'], rehearsal=True)
    with pytest.raises(pf.Refused, match='real root'):
        pf.contain(c['root'], [r'D:\jepa_phase0\runs\cr_probe_x'])
    assert pf.contain(c['root'], [r1]) == 0


@pytest.mark.skipif(shutil.which('powershell') is None, reason='Windows PowerShell not available')
def test_ps1_rehearsal_never_unseals_a_surrogate_outside_the_fixture(surrogate, tmp_path):
    a, _, foreign = surrogate
    out, mirror = tmp_path / 'out', tmp_path / 'm'
    proc = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           os.path.join(REPO, 'autopilot', 'cr_postfreeze.ps1'), '-Rehearsal',
                           '-FixtureRoot', a['root'], '-OutRoot', str(out), '-TargetRepo',
                           str(mirror), '-FreshMirror', '-NBoot', '10', '-SkipPoster',
                           '-SkipReceiptPreview'], capture_output=True, text=True, timeout=600)
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, text[-3000:]
    assert 'outside the fixture root' in text and '--allow-before-freeze' not in text
    assert not os.path.exists(os.path.join(foreign, 'unsealed_results.json'))
    assert not any(os.path.exists(os.path.join(a['runs_root'], d, 'unsealed_results.json'))
                   for d in os.listdir(a['runs_root']))
    assert not mirror.exists()  # the plan runs before anything is copied


# ---------------------------------------------------------------------------
# V8-2: cache key and final consistency follow the completion evidence
# ---------------------------------------------------------------------------

def test_missing_or_changed_stop_file_forces_recomputation_and_flags_stale_final(tmp_path):
    c = pfx.build_campaign(tmp_path / 'c', **SMALL)
    plan = plan_of(c)
    for run_dir in plan['to_unseal']:
        cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    plan_path = _write(tmp_path / 'plan.json', plan_of(c))
    final_path = str(tmp_path / 'final.json')
    cr_stats.cmd_final(plan['ready_dirs'], n_boot=30, out=final_path, allow_before_freeze=True)
    key0 = pf.input_key(plan_path, [])
    assert pf.check_final(plan_path, final_path) == 0
    stop = os.path.join(c['runs_root'], 'cr_seed_v1_envelope_s9012', 'stop_epoch_050.json')
    good = open(stop, 'rb').read()
    for label, write in (('missing', lambda: os.remove(stop)),
                         ('changed', lambda: open(stop, 'wb').write(good.replace(
                             b'2026-10-20T12:00:00-07:00', b'2026-10-22T09:30:00-07:00')))):
        write()
        now = plan_of(c)
        assert not now['seed_indices']['9012']['primary_on_time'], label
        plan_path = _write(tmp_path / ('plan_%s.json' % label), now)
        assert pf.input_key(plan_path, []) != key0, label  # cached final is not reused
        assert pf.check_final(plan_path, final_path) == 2, label  # stale final is refused
        fresh = str(tmp_path / ('final_%s.json' % label))
        cr_stats.cmd_final(now['ready_dirs'], n_boot=30, out=fresh, allow_before_freeze=True)
        assert pf.check_final(plan_path, fresh) == 0, label
        assert json.load(open(fresh))['analysis'][cr_stats.PRIMARY_VARIANT][
            'complete_seed_indices'] == [1234, 5678]
        with open(stop, 'wb') as stream:
            stream.write(good)
    assert pf.input_key(_write(tmp_path / 'plan_back.json', plan_of(c)), []) == key0


# ---------------------------------------------------------------------------
# V10-1: an edited final is refused by check-final; cache records bind the output digest
# ---------------------------------------------------------------------------

def test_check_final_refuses_edited_labels_and_values(tmp_path):
    c = pfx.build_campaign(tmp_path / 'c', seeds=(1234, 5678), **SMALL)
    plan = plan_of(c)
    for run_dir in plan['to_unseal']:
        cr_stats.cmd_unseal(run_dir, allow_before_freeze=True)
    plan_path = _write(tmp_path / 'plan.json', plan_of(c))
    final_path = tmp_path / 'final.json'
    cr_stats.cmd_final(plan['ready_dirs'], n_boot=30, out=str(final_path),
                       allow_before_freeze=True)
    assert pf.check_final(plan_path, str(final_path)) == 0
    good = final_path.read_text()
    block = lambda f: f['analysis'][cr_stats.PRIMARY_VARIANT]  # noqa: E731

    def relabel(f):
        for o in block(f)['outcomes'].values():
            o['label'] = 'reversed' if o['label'] != 'reversed' else 'consistent_direction'

    for edit in (relabel,
                 lambda f: block(f)['primary_contrasts'][0].update(delta=0.25),
                 lambda f: [o.update(random_range=1e-5) for o in block(f)['outcomes'].values()]):
        f = json.loads(good)
        edit(f)
        final_path.write_text(json.dumps(f))
        assert pf.check_final(plan_path, str(final_path)) == 2
    final_path.write_text(good)
    assert pf.check_final(plan_path, str(final_path)) == 0


@pytest.mark.skipif(shutil.which('powershell') is None, reason='Windows PowerShell not available')
def test_ps1_cache_record_binds_the_output_digest(tmp_path):
    script = os.path.join(REPO, 'autopilot', 'cr_postfreeze.ps1')
    text = open(script, encoding='utf-8').read()
    for output in ('final.json', 'md_test.json'):  # both cached analyses use the record
        assert len(re.findall(r"Test-CacheValid \(Join-Path \$OutRoot '%s'\)" % output, text)) == 1
        assert len(re.findall(r"Save-CacheRecord \(Join-Path \$OutRoot '%s'\)" % output, text)) == 1
    out, key = tmp_path / 'final.json', tmp_path / 'final.key'
    out.write_text('{"label": "reversed"}')
    driver = tmp_path / 'cache.ps1'
    driver.write_text("\n".join([
        "$ErrorActionPreference = 'Stop'",
        ". '%s' -LibraryOnly" % script,
        "$o = '%s'; $k = '%s'" % (out, key),
        "Save-CacheRecord $o $k 'abc'",
        "Write-Output ('fresh ' + (Test-CacheValid $o $k 'abc'))",
        "Write-Output ('otherkey ' + (Test-CacheValid $o $k 'abd'))",
        "Set-Content -Path $o -Value '{\"label\": \"consistent_direction\"}'",
        "Write-Output ('edited ' + (Test-CacheValid $o $k 'abc'))",
        "Set-Content -Path $k -Value 'abc' -Encoding ascii",
        "Write-Output ('keyonly ' + (Test-CacheValid $o $k 'abc'))", ""]))
    proc = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           str(driver)], capture_output=True, text=True, timeout=120)
    lines = proc.stdout
    assert proc.returncode == 0, lines + proc.stderr
    for expected in ('fresh True', 'otherkey False', 'edited False', 'keyonly False'):
        assert expected in lines, lines
    assert 'changed after it was computed' in lines


def test_scratch_poster_copy_brings_the_figure_directories_it_names(tmp_path):
    """The poster names figure directories by relative path (graphicspath); its scratch build
    needs read-only copies of them at the same relative place (the paper comes from the
    scratch paper copy)."""
    repo = tmp_path / 'repo'
    poster = repo / 'poster' / 'genai4health2026'
    figs = repo / 'autopilot' / 'figs' / 'poster'
    for d in (poster, figs, repo / 'paper' / 'genai4health2026' / 'figures'):
        d.mkdir(parents=True)
    (poster / 'poster.tex').write_text(
        '\\graphicspath{{../../autopilot/figs/poster/}{../../paper/genai4health2026/figures/}}\n'
        '\\includegraphics{a.pdf}\n')
    (figs / 'a.pdf').write_bytes(b'%PDF-1.4 synthetic')
    dst = tmp_path / 'out' / 'scratch' / 'poster' / 'genai4health2026'
    for _ in range(2):  # re-runnable
        pf.scratch_copy(poster, dst, 'poster')
        assert (tmp_path / 'out' / 'scratch' / 'autopilot' / 'figs' / 'poster' / 'a.pdf').is_file()
    assert not (tmp_path / 'out' / 'scratch' / 'paper').exists()
    assert pf.poster_external_dirs(poster) == [figs.resolve()]


def test_ps1_always_runs_the_md_secondary_analysis_and_passes_it_to_build():
    """H8: md-test runs cr_md_regression on the test split (no skip switch), build always gets
    --md, and the real run checks the MD metadata before anything is unsealed."""
    text = open(os.path.join(REPO, 'autopilot', 'cr_postfreeze.ps1'), encoding='utf-8').read()
    step = text[text.index("Invoke-Step 'md-test'"):text.index("Invoke-Step 'cb-audit'")]
    assert "cr_md_regression.py" in step and "'--split', 'test'" in step
    assert "'--metadata', $MdMetadata" in step and 'return' in step  # only the cache skip
    build_args = text[text.index('function Get-BuildArgs'):text.index('function Read-BuildArgs')]
    assert "'--md', (Join-Path $OutRoot 'md_test.json')" in build_args
    assert 'SkipMd' not in text
    preflight = text[text.index("Invoke-Step 'preflight'"):text.index("Invoke-Step 'inventory'")]
    assert preflight.index('MD metadata missing') < preflight.index("'plan'")


# ---------------------------------------------------------------------------
# V8-5: fixtures are time-invariant
# ---------------------------------------------------------------------------

def test_fixture_eligibility_does_not_depend_on_the_generation_clock(tmp_path, monkeypatch):
    import datetime
    after = datetime.datetime.fromisoformat('2026-10-22T15:00:01+00:00')
    monkeypatch.setattr(cr_stats, '_now', lambda: after)
    monkeypatch.setattr(pfx.evaluation, '_now_iso', lambda: '2026-10-22T08:00:01-07:00')
    plain = pfx.build_campaign(tmp_path / 'plain', seeds=(1234,), **SMALL)
    plan = plan_of(plain)
    assert plan['late'] == [] and plan['seed_indices']['1234']['primary_on_time']
    assert all(r['eligibility']['eligible'] for r in plan['runs'] if r['status'] == 'ready')
    late = pfx.build_campaign(tmp_path / 'late', seeds=(1234, 5678),
                              late=[('envelope', 5678)], **SMALL)
    assert plan_of(late)['late'] == ['E2']
    d = pfx.fx.build_design(tmp_path / 'design', 'A', 'A', 'M+', n_test=200, n_boot=30)
    block = json.load(open(d['final']))['analysis'][cr_stats.PRIMARY_VARIANT]
    assert block['complete_seed_indices'] == [1234, 5678]


# ---------------------------------------------------------------------------
# V8-6: the p13 step and its Tectonic child run at Idle
# ---------------------------------------------------------------------------

PRIORITY_PROBE = """import ctypes, subprocess, sys
GET = ('import ctypes; k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p; '
       'k.GetPriorityClass.argtypes = [ctypes.c_void_p]; '
       'cls = k.GetPriorityClass(k.GetCurrentProcess())')
exec(GET)
print('child', cls)
sys.stdout.flush()
subprocess.run([sys.executable, '-c', GET + '; print("grandchild", cls)'])
"""


@pytest.mark.skipif(shutil.which('powershell') is None, reason='Windows PowerShell not available')
def test_p13_step_and_its_compiler_run_at_idle(tmp_path):
    script = os.path.join(REPO, 'autopilot', 'cr_postfreeze.ps1')
    text = open(script, encoding='utf-8').read()
    calls = re.findall(r"Invoke-Py 'p13_camera_ready'[^\n]*", text)
    assert len(calls) == 1 and "-Priority 'Idle'" in calls[0]
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    probe = tmp_path / 'probe.py'
    probe.write_text(PRIORITY_PROBE)
    driver = tmp_path / 'driver.ps1'
    driver.write_text("\n".join([
        "$ErrorActionPreference = 'Stop'",
        # Start above Idle (pytest itself may run at Idle), so Idle can only come from the request.
        "(Get-Process -Id $PID).PriorityClass = 'BelowNormal'",
        ". '%s' -LibraryOnly" % script,
        "$LogDir = '%s'; $Phase = 'test'; $RepoRoot = '%s'; $Python = '%s'"
        % (log_dir, REPO, sys.executable),
        "$c = Invoke-Py 'priority_probe' @('%s') -Priority 'Idle'" % probe,
        "Write-Output ('orchestrator after: ' + (Get-Process -Id $PID).PriorityClass)",
        "exit $c", ""]))
    proc = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           str(driver)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = open(log_dir / 'test_priority_probe.log', encoding='utf-8-sig').read()
    assert 'PRIORITY: Idle' in log, log
    assert 'child 64' in log and 'grandchild 64' in log, log  # IDLE_PRIORITY_CLASS = 0x40
    # the orchestrator's own class is restored after the start
    assert 'orchestrator after: BelowNormal' in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# receipt migration (coordinator note 3; V8-3)
# ---------------------------------------------------------------------------

MAIN_OLD = (
    # P1: changed only by an integration passage; its literals keep their claim text.
    "The replication protocol stops after the epoch-50 checkpoint of every run, with 75 linear "
    "heads trained in total for the probe stage. It adds two further continuations each of the "
    "guided strategies and keeps the data order.\n\n"
    # P2: V8-3 counterexample, another arm and split (not an integration change).
    "For the original RANDOM encoder evaluated on the validation split, using the frozen linear "
    "probe and the common pooling setup, the AUC was 0.8641 with the uncertainty reported "
    "separately in the supplementary results.\n\n"
    # P3: changed qualifier outside the literal's window.
    "Under the primary pooling with mean features, measured over the head seeds of both "
    "strategies, the reported AUC difference was 0.0047 in the comparison at the final epoch.\n\n"
    # P4: the integration replaces the ranking sentence (its epoch-100 literal disappears).
    + r"\ArmBest{} has the highest epoch-100 point estimate, exceeding \textsc{envelope} by "
    r"$\DOracleEnvelopeEpHundred$ (CI $\DOracleEnvelopeEpHundredCI$). The paired intervals "
    "include zero at epochs 50 and 75 of the original comparison runs.\n\n"
    # P6 / P7: identical +-24-character windows; P6 changed by the integration, P7 otherwise.
    + "Our summary lists results at epochs 30 and 40 for every policy, including the context they "
    r"leave visible. \CRSeedResultsPending" + "\n\n"
    "The second summary lists results at epochs 30 and 40 for every policy, measured on the "
    "original arm.\n")
MAIN_NEW = (
    "The replication protocol stops after the epoch-50 checkpoint of every run, with 75 linear "
    "heads trained in total for the probe stage. It adds \\RepNRegisteredSeedsWord{} further "
    "continuations each of the guided strategies and keeps the data order.\n\n"
    "For the new ENVELOPE encoder evaluated on the held-out test split, using the frozen linear "
    "probe and the common pooling setup, the AUC was 0.8641 with the uncertainty reported "
    "separately in the supplementary results.\n\n"
    "Under the patch-max pooling with max features, measured over the head seeds of both "
    "strategies, the reported AUC difference was 0.0047 in the comparison at the final epoch.\n\n"
    + r"In the original single runs, the \ArmBest{}--\textsc{envelope} difference at the final "
    r"evaluated epoch is $\DOracleEnvelopeEpHundred$ (CI $\DOracleEnvelopeEpHundredCI$). The "
    "paired intervals include zero at epochs 50 and 75 of the original comparison runs.\n\n"
    + "Our summary lists results at epochs 30 and 40 for every policy, including the context they "
    r"leave visible. \RepConclusion{}" + "\n\n"
    "The second summary lists results at epochs 30 and 40 for every policy, measured on the "
    "replicated arm.\n")
# P5: a literal inside the replaced passage reappears in its replacement (human acceptance).
PROTO_OLD = ("Each arm continues from the shared epoch-25 checkpoint with two new seeds, which set "
             "the data order.\n")
PROTO_NEW = ("Each arm continues from the shared epoch-25 checkpoint with "
             "\\RepNRegisteredSeedsWord{} new seeds (the third added by a registered amendment), "
             "which set the data order.\n")


def _receipt_case(tmp_path, crlf=True):
    old_dir, paper, results = tmp_path / 'old', tmp_path / 'paper', tmp_path / 'results'
    for d in (old_dir, paper, results):
        d.mkdir()
    docs = {'main_submission.tex': (MAIN_OLD, MAIN_NEW), 'compact_protocol.tex': (PROTO_OLD, PROTO_NEW)}
    old_rows, new_rows = [], []
    for rel, (old, new) in docs.items():
        (old_dir / rel).write_text(old, encoding='utf-8')
        (paper / rel).write_text(new, encoding='utf-8')
        old_rows += [r for r in numeric.literals(old, rel) if r['status'] != 'structural']
        new_rows += [r for r in numeric.literals(new, rel) if r['status'] != 'structural']
    literals = [{'file': r['file'], 'context_sha256': r['context_sha256'],
                 'token_index': r['token_index'], 'value': r['value'],
                 'review': {'kind': 'protocol', 'reviewer': 'original reviewer',
                            'rationale': 'claim %s' % r['value'], 'evidence': []}}
                for r in old_rows]
    review = {'version': 1, 'sources': {}, 'macros': {}, 'literals': literals, 'figures': []}
    text = json.dumps(review, indent=2) + '\n'
    (paper / 'numeric_reviews.json').write_bytes((text.replace('\n', '\r\n') if crlf else text)
                                                  .encode('utf-8'))
    report = {'errors': ["stale or inapplicable literal review (context/token changed): "
                         "('%s', '%s', %d)" % (r['file'], r['context_sha256'], r['token_index'])
                         for r in old_rows],
              'items': [dict(r, status='unresolved') for r in new_rows]}
    (tmp_path / 'p15.json').write_text(json.dumps(report))
    # Results the paper was integrated with: scenario B with a figure (ranking sentence restated).
    (results / 'cr_results.values.json').write_text(json.dumps(
        {'scenario': {'scenario': 'B'}, 'figure': {'png': 'x', 'pdf': 'y'}}))
    (results / 'cr_results.tex').write_text('% Scenario: B; labels ...\n')
    key = lambda r: (r['file'], r['value'], r['context'][:12])  # noqa: E731
    return old_dir, paper, results, {key(r): r for r in old_rows}, {key(r): r for r in new_rows}


def _keys(paper):
    data = json.loads((paper / 'numeric_reviews.json').read_bytes().decode('utf-8'))
    return {(e['file'], e['context_sha256'], e['token_index']): e for e in data['literals']}


def _k(row):
    return (row['file'], row['context_sha256'], row['token_index'])


def test_receipt_migration_only_follows_the_integration_mapping(tmp_path):
    old_dir, paper, results, old, new = _receipt_case(tmp_path)
    base = ['--paper-dir', str(paper), '--old-dir', str(old_dir), '--results-dir', str(results),
            '--report', str(tmp_path / 'p15.json'), '--out', str(tmp_path / 'plan.json')]
    assert mig.main(base) == 0  # plan only
    plan = json.load(open(tmp_path / 'plan.json'))
    moved = {(m['old'][0], m['value'], m['new'][1]) for m in plan['migrations']}
    m = 'main_submission.tex'
    expected = {(m, '-50', new[(m, '-50', 'The replicat')]['context_sha256']),
                (m, '75', new[(m, '75', 'The replicat')]['context_sha256']),
                (m, '50', new[(m, '50', 'In the origi')]['context_sha256']),
                (m, '75', new[(m, '75', 'In the origi')]['context_sha256']),
                (m, '30', new[(m, '30', 'Our summary ')]['context_sha256']),
                (m, '40', new[(m, '40', 'Our summary ')]['context_sha256'])}
    assert moved == expected  # P1, P4 (outside the replaced sentence) and P6 only
    manual = {(i['key'][0], i['value'], i['new_context'][:12]): i for i in plan['manual']}
    assert set(manual) == {(m, '0.8641', 'For the new '), (m, '0.0047', 'Under the pa'),
                           (m, '30', 'The second s'), (m, '40', 'The second s'),
                           ('compact_protocol.tex', '-25', 'Each arm con')}
    # V8-3: same value and window, but another arm / split / qualifier: never automatic.
    for k in ((m, '0.8641', 'For the new '), (m, '0.0047', 'Under the pa'),
              (m, '30', 'The second s')):
        sug = manual[k]['suggested']
        assert sug and 'outside the integration mapping' in sug['why_not_automatic']
        assert sug['old_context'] != manual[k]['new_context']
    # Duplicate windows: P7's predecessor is P7's old receipt, never P6's.
    assert manual[(m, '30', 'The second s')]['suggested']['old'][1] == \
        old[(m, '30', 'The second s')]['context_sha256']
    assert 'replaced' in manual[('compact_protocol.tex', '-25', 'Each arm con')]['suggested'][
        'why_not_automatic']
    assert [o['value'] for o in plan['obsolete_dropped']] == ['-100']
    assert mig.main(base + ['--apply']) == 0
    keys = _keys(paper)
    for item in plan['migrations']:
        assert tuple(item['new']) in keys and tuple(item['old']) not in keys
    p2_old, p2_new = old[(m, '0.8641', 'For the orig')], new[(m, '0.8641', 'For the new ')]
    assert _k(p2_new) not in keys and _k(p2_old) in keys  # the changed claim keeps no receipt
    assert _k(old[(m, '-100', '\\ArmBest{} h')]) not in keys
    raw = (paper / 'numeric_reviews.json').read_bytes()
    assert b'\r\n' in raw and b'\n' not in raw.replace(b'\r\n', b'')
    # A human may accept after comparing old_context and new_context (named reviewer, exact id).
    report = json.load(open(tmp_path / 'p15.json'))
    report['errors'] = [e for e in report['errors'] if p2_old['context_sha256'] in e]
    report['items'] = [i for i in report['items'] if i['value'] == '0.8641']
    (tmp_path / 'p15.json').write_text(json.dumps(report))
    pid = manual[(m, '0.8641', 'For the new ')]['id']
    assert mig.main(base + ['--apply', '--accept', pid]) == 2  # no reviewer
    assert mig.main(base + ['--apply', '--accept', 'main_submission.tex:0000:9',
                            '--reviewer', 'Human']) == 2
    assert mig.main(base + ['--apply', '--accept', pid, '--reviewer', 'Human']) == 0
    entry = _keys(paper)[_k(p2_new)]
    assert 'checked against the new context by Human' in entry['review']['reviewer']


def test_receipt_migration_without_results_dir_maps_nothing(tmp_path):
    old_dir, paper, _, _, _ = _receipt_case(tmp_path)
    assert mig.main(['--paper-dir', str(paper), '--old-dir', str(old_dir), '--report',
                     str(tmp_path / 'p15.json'), '--out', str(tmp_path / 'plan.json'),
                     '--apply']) == 0
    plan = json.load(open(tmp_path / 'plan.json'))
    assert plan['migrations'] == [] and plan['obsolete_dropped'] == []
    assert plan['automatic_mapping'] is False and len(plan['manual']) == 11


def test_receipt_migration_refuses_the_real_review_file_without_permission(tmp_path):
    old_dir, paper, results, _, _ = _receipt_case(tmp_path)
    before = assets.sha256(assets.PAPER / 'numeric_reviews.json')
    args = ['--paper-dir', str(paper), '--old-dir', str(old_dir), '--results-dir', str(results),
            '--report', str(tmp_path / 'p15.json'), '--out', str(tmp_path / 'plan.json'),
            '--apply', '--review-file', str(assets.PAPER / 'numeric_reviews.json')]
    assert mig.main(args) == 2
    assert mig.main(['--paper-dir', str(paper), '--old-dir', str(paper), '--report',
                     str(tmp_path / 'p15.json'), '--out', str(tmp_path / 'plan.json')]) == 2
    assert assets.sha256(assets.PAPER / 'numeric_reviews.json') == before
    assert re.search(r'stale', mig.__doc__)
