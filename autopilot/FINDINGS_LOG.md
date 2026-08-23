# FINDINGS LOG

One entry per milestone. Records new probes and any headline contrast that moved, so nothing changes in the paper without being surfaced here first.

## baseline-before-phase-B  -  2026-08-23T01:33:05-07:00

**New probes since last milestone:**

| probe | test AUC |
|---|---|
| `anatomy-v1@ep30@fp32` | 0.858274 |
| `anatomy-v2@ep35@fp32` | 0.866129 |
| `anatomy-v2@ep40@fp32` | 0.868251 |
| `anatomy-v2@ep50@fp32` | 0.865386 |
| `ancestor@ep25@fp32` | 0.848680 |
| `cover-f021@ep27@fp32` | 0.848347 |
| `cover-f021@ep30@fp32` | 0.852249 |
| `cover-f021@ep34@fp32` | 0.857083 |
| `cover-f021@ep50@fp32` | 0.864281 |
| `envelope@ep100@fp16` | 0.880743 |
| `envelope@ep100@fp32` | 0.880761 |
| `envelope@ep30@fp32` | 0.853917 |
| `envelope@ep50@fp16` | 0.876064 |
| `envelope@ep50@fp32` | 0.876063 |
| `envelope@ep75@fp16` | 0.880307 |
| `oracle@ep100@fp16` | 0.885485 |
| `oracle@ep100@fp32` | 0.885293 |
| `oracle@ep50@fp16` | 0.874030 |
| `oracle@ep75@fp16` | 0.883636 |
| `random@ep100@fp16` | 0.874581 |
| `random@ep100@fp32` | 0.874485 |
| `random@ep50@fp16` | 0.864097 |
| `random@ep75@fp16` | 0.872302 |

No headline contrast moved by more than 5e-5.

**Current headline contrasts:**

| contrast | delta AUC |
|---|---|
| envelope-random@ep100 | +0.006276 |
| envelope-random@ep50 | +0.011967 |
| envelope-random@ep75 | +0.008005 |
| oracle-random@ep100 | +0.010808 |
| oracle-random@ep50 | +0.009933 |
| oracle-random@ep75 | +0.011333 |


## phase-A  -  2026-08-23T04:13:46-07:00

**New probes since last milestone:**

| probe | test AUC |
|---|---|
| `cover-f021@ep73@fp32` | 0.864717 |
| `envelope@ep75@fp32` | 0.880305 |
| `oracle@ep50@fp32` | 0.874015 |
| `random@ep50@fp32` | 0.864121 |

No headline contrast moved by more than 5e-5.

**Current headline contrasts:**

| contrast | delta AUC |
|---|---|
| envelope-random@ep100 | +0.006276 |
| envelope-random@ep50 | +0.011942 |
| envelope-random@ep75 | +0.008005 |
| oracle-random@ep100 | +0.010808 |
| oracle-random@ep50 | +0.009894 |
| oracle-random@ep75 | +0.011333 |

