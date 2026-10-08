"""cr_seed_v1 config generator: declared-key diffs, invariants, resume configs."""
import copy

import pytest
import yaml

from scripts import cr_make_configs as mk


@pytest.fixture(scope="module")
def generated():
    report, texts = mk.generate(write=False)
    return report, texts


def test_generation_has_no_errors_and_matches_disk(generated):
    report, texts = generated
    assert report["errors"] == []
    assert len(report["configs"]) == 6
    for path, text in texts.items():
        assert path.exists(), path
        assert path.read_text(encoding="utf-8") == text, "stale generated file %s" % path


def test_diffs_only_declared_keys(generated):
    report, _ = generated
    for name, rec in report["configs"].items():
        assert rec["undeclared"] == [], name
        decl = set(mk.flatten_decl(mk.declared_keys(rec["arm"], rec["seed"])))
        touched = set(rec["changed"]) | set(rec["added"])
        assert touched <= decl, (name, touched - decl)
        assert set(rec["removed"]) <= set(mk.ARM_REMOVALS[rec["arm"]])


def test_check_declared_flags_an_undeclared_change():
    archived = mk.load_yaml(mk.ARCHIVED["centroid"])
    cfg = mk.build_config("centroid", 1234, archived)
    cfg["mask"]["pred_mask_scale"] = [0.1, 0.2]
    cfg["mask"]["curriculum"]["oracle_region_frac"] = 0.3
    chk = mk.check_declared("centroid", 1234, archived, cfg)
    assert chk["undeclared"] == ["mask.curriculum.oracle_region_frac", "mask.pred_mask_scale"]


def test_seed_and_arm_pairs(generated):
    report, _ = generated
    for arm, keys in report["seed_pair_diffs"].items():
        assert keys == ["logging.folder", "logging.write_tag", "meta.seed"], arm
    for pair, keys in report["arm_pair_diffs"].items():
        assert all(k.startswith("mask.curriculum") or k.startswith("logging.") for k in keys), pair


@pytest.mark.parametrize("arm", mk.ARMS)
@pytest.mark.parametrize("seed", mk.SEEDS)
def test_generated_invariants(arm, seed):
    cfg = yaml.safe_load(mk.config_path(arm, seed).read_text(encoding="utf-8"))
    mk.validate_run_config(cfg, arm, seed)
    assert cfg["optimization"]["epochs"] == 100
    assert cfg["optimization"]["accum_steps"] == 8 and cfg["data"]["batch_size"] == 64
    assert cfg["data"]["num_workers"] == mk.NUM_WORKERS == 4
    assert cfg["data"]["prefetch_factor"] == 2
    assert cfg["data"]["slice_cache_dir"] == r"C:\jepa_data\slice_cache"
    assert cfg["meta"]["amp_target"] is False and cfg["meta"]["use_bfloat16"] is False
    assert cfg["meta"]["resume_policy"] == "fork" and cfg["meta"]["fork_start_epoch"] == 25
    assert cfg["meta"]["read_checkpoint"] == mk.ANCESTOR
    assert cfg["optimization"]["save_every"] == 5 and cfg["optimization"]["patience"] == 9999
    assert cfg["optimization"]["lr"] == 0.00025 and cfg["optimization"]["warmup"] == 5
    assert cfg["optimization"]["ema"] == [0.996, 1.0] and cfg["optimization"]["ipe_scale"] == 1.0
    assert cfg["logging"]["folder"] == r"D:\jepa_phase0\runs\cr_seed_v1_%s_s%d" % (arm, seed)
    assert cfg["logging"]["write_tag"] == "jepa_patch_cr_seed_v1_%s_s%d" % (arm, seed)
    assert "pred_target_k" not in cfg["mask"]
    cur = cfg["mask"].get("curriculum") or {}
    if arm == "centroid":
        assert cur["oracle_lateral_frac"] == 0.6 and cur["enc_truncate"] == "prefix"
    if arm == "envelope":
        assert cur["mirage_overlap_fallback"] == "legacy_uniform_v1"
        assert cur["mirage_guide_dir"] == r"D:\jepa_phase0\fairvision-glaucoma\mirage_guides"
        assert cur["enc_truncate"] == "prefix"
    if arm == "random":
        assert cur == {"enabled": False}


@pytest.mark.parametrize("key,value", [
    ("optimization.epochs", 50), ("meta.amp_target", True), ("optimization.accum_steps", 2),
    ("meta.fork_start_epoch", 27), ("meta.read_checkpoint", r"D:\x\resume-ep27.pth.tar"),
    ("mask.pred_target_k", 16), ("meta.seed", 1)])
def test_validate_rejects(key, value):
    cfg = mk.build_config("envelope", 1234)
    mk.set_key(cfg, key, value)
    with pytest.raises(ValueError):
        mk.validate_run_config(cfg, "envelope", 1234)


def test_envelope_requires_legacy_flag():
    cfg = mk.build_config("envelope", 5678)
    cfg["mask"]["curriculum"]["mirage_overlap_fallback"] = "least_overlap_v2"
    with pytest.raises(ValueError):
        mk.validate_run_config(cfg, "envelope", 5678)


def test_resume_config_is_exact_without_fork_fields():
    fork = mk.build_config("centroid", 1234)
    last = r"D:\jepa_phase0\runs\cr_seed_v1_centroid_s1234\jepa_patch_cr_seed_v1_centroid_s1234-last.pth.tar"
    res = mk.make_resume_config(fork, last)
    assert res["meta"]["resume_policy"] == "exact"
    assert "fork_start_epoch" not in res["meta"]
    assert res["meta"]["read_checkpoint"] == last and res["meta"]["load_checkpoint"] is True
    mk.validate_run_config(res, "centroid", 1234, expect_fork=False)
    d = mk.diff_flat(fork, res)
    assert set(d["changed"]) == {"meta.resume_policy", "meta.read_checkpoint"}
    assert set(d["removed"]) == {"meta.fork_start_epoch"} and not d["added"]
    assert fork["meta"]["resume_policy"] == "fork"  # input untouched


def test_random_cb_is_a_placeholder():
    with pytest.raises(NotImplementedError):
        mk.build_config("random_cb", 1234)


def test_archived_inputs_are_untouched_copies():
    archived = mk.load_yaml(mk.ARCHIVED["envelope"])
    before = copy.deepcopy(archived)
    mk.build_config("envelope", 1234, archived)
    assert archived == before
    assert archived["meta"]["read_checkpoint"].endswith("resume-ep27.pth.tar")
