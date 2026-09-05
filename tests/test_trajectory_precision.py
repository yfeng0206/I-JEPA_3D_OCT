import pytest

from autopilot.p8_make_assets import matched_trajectory_keys


@pytest.mark.parametrize("arm", ["envelope", "oracle", "cover-f021"])
@pytest.mark.parametrize("epoch", [50, 75, 100])
def test_trajectory_null_matches_arm_precision(arm, epoch):
    precision = "fp32" if arm == "cover-f021" else "fp16"
    source = f"{arm}@ep{epoch}@{precision}"
    table = {source: {}, f"random@ep{epoch}@fp16": {}, f"random@ep{epoch}@fp32": {}}
    assert matched_trajectory_keys(arm, epoch, table) == (
        source, f"random@ep{epoch}@{precision}"
    )


def test_trajectory_never_falls_back_to_a_different_precision():
    with pytest.raises(ValueError, match="random@ep50@fp16"):
        matched_trajectory_keys(
            "envelope", 50, {"envelope@ep50@fp16": {}, "random@ep50@fp32": {}}
        )


def test_trajectory_requires_the_guided_measurement():
    with pytest.raises(ValueError, match="oracle@ep100@fp16"):
        matched_trajectory_keys("oracle", 100, {"random@ep100@fp16": {}})
