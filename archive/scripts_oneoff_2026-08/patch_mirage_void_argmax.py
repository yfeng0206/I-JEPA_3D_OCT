"""Implement the void/ignore-class exclusion that MIRAGE documents but never
implemented.

At `run_seg_tuning.py` the evaluation loop carries this comment:

    # If there is void, exclude it from the preds and take second highest class
    seg_pred_argmax = seg_pred.argmax(dim=1)

The comment describes the intended behaviour; the code does a plain
unconstrained argmax over ALL logits, including the ignore/void channel.

Why that matters for merged multi-dataset training
--------------------------------------------------
Our merged taxonomy declares four label indices (Elsewhere / InnerRetina /
Choroid / Background) where Background is the ignore index, so the model emits
a logit for a class that is never a valid target.

`compute_metrics` reduces per-class IoU with `np.nanmean`.  If the void class
is never predicted, its IoU is 0/0 = NaN and `nanmean` correctly drops it.  The
moment ONE pixel is predicted as void, its union becomes non-zero while its
intersection stays zero, so IoU becomes 0.0 -- a real number that `nanmean`
now averages in.  Validation mIoU is then silently deflated by roughly a
quarter, which both misreports quality and corrupts best-checkpoint selection,
because the deflation depends on how often the void class happens to win the
argmax in that particular epoch.

A 1-epoch smoke did not trigger it (reported mIoU 82.14 is inconsistent with a
void IoU of 0 being included), but the risk is latent across a 200-epoch run,
so we close it rather than hope.

Fix: suppress the ignore channel before the argmax, which is exactly the
"exclude it from the preds and take second highest class" the comment asks for.
Idempotent.
"""
from __future__ import annotations

import ast
import pathlib
import sys

TARGET = pathlib.Path(r'D:\jepa_phase0\mirage-goals\MIRAGE\run_seg_tuning.py')

ANCHOR = """        # If there is void, exclude it from the preds and take second highest class
        seg_pred_argmax = seg_pred.argmax(dim=1)"""

PATCH = """        # If there is void, exclude it from the preds and take second highest class
        # (implemented: the original code did a plain argmax over all logits,
        # so the void/ignore channel could win and turn that class's IoU from
        # NaN -- dropped by nanmean -- into 0.0, deflating mIoU and corrupting
        # best-checkpoint selection.)
        if ignore_index is not None and 0 <= ignore_index < seg_pred.shape[1]:
            seg_pred = seg_pred.clone()
            seg_pred[:, ignore_index] = float('-inf')
        seg_pred_argmax = seg_pred.argmax(dim=1)"""

MARKER = "seg_pred[:, ignore_index] = float('-inf')"


def main() -> int:
    text = TARGET.read_text(encoding='utf-8')
    if MARKER in text:
        print('already patched')
    else:
        if text.count(ANCHOR) != 1:
            raise SystemExit('FAILED: anchor found %d times, expected 1'
                             % text.count(ANCHOR))
        TARGET.write_text(text.replace(ANCHOR, PATCH, 1), encoding='utf-8')
        print('patched void-class exclusion')

    src = TARGET.read_text(encoding='utf-8')
    ast.parse(src)
    print('syntax OK')

    # ignore_index must actually be a parameter of the enclosing function.
    idx = src.index(MARKER)
    head = src.rfind('\ndef ', 0, idx)
    sig = src[head:src.index(':\n', head)]
    print('enclosing function has ignore_index param:',
          'ignore_index' in sig)
    return 0


if __name__ == '__main__':
    sys.exit(main())
