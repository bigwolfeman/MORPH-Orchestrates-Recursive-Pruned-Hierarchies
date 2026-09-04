"""Sustained detonation tripwire for CONSTRAINED slot-loop arms (arc E2, 2026-09-04).

The single-step rule (tulfm/tripwire.py: preclip/total > 1e4 at step >= 200) was
validated on unconstrained runs. Under the every-iteration gain hinge a one-step
gradient outlier snaps back within ~7 steps (to-mnext-y2-iter-all: 17034 at 1684, then
8, 10, 10, 9, 11, 7, 2.6), so the single-step rule killed a healthy run. Calibrated on
the real detonation (to-mnext-y2-iter: 81617 at 2556, 116930 at 2557, then 26754 at 2567
and 27698 at 2586 with a baseline shifted from ~50 to hundreds):

  DETONATED  iff  any row > 1e5  OR  >= 2 rows > 1e4 within any 40-step window   (step >= 200)
  EXCURSION  iff  exactly one row > 1e4 (and nothing above)                       exit 3
  HEALTHY    otherwise                                                            exit 0

Usage / exit codes as tripwire.py: 4 detonated, 3 excursion/ambiguous, 0 healthy, 2 unreadable.
"""
import json
import sys


def main() -> int:
    path = sys.argv[1]
    hi = []            # steps with preclip/total > 1e4
    mx, mx_step, last = 0.0, None, 0
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = int(r.get("step", 0))
                last = max(last, s)
                v = r.get("preclip/total")
                if v is None or s < 200:
                    continue
                v = float(v)
                if v > mx:
                    mx, mx_step = v, s
                if v > 1e5:
                    print(f"DETONATED step={s} max={mx:.3g}@{mx_step} (single row > 1e5)")
                    return 4
                if v > 1e4:
                    hi.append(s)
                    if len(hi) >= 2 and s - hi[-2] <= 40:
                        print(f"DETONATED step={s} max={mx:.3g}@{mx_step} (2 rows > 1e4 within 40 steps: {hi[-2]}, {s})")
                        return 4
    except FileNotFoundError:
        print("UNREADABLE (no probe file yet)")
        return 2
    if hi:
        print(f"EXCURSION last={last} max={mx:.3g}@{mx_step} (one-step outliers at {hi}, recovered)")
        return 3
    if mx > 1e3:
        print(f"AMBIGUOUS last={last} max={mx:.3g}@{mx_step}")
        return 3
    print(f"HEALTHY last={last} max={mx:.3g}@{mx_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
