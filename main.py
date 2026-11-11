from datetime import datetime
from pathlib import Path
import numpy as np
from fuzzy_fusion import MamdaniFusion, reliability_from_validation, ds_fuse, CLASSES, MODALITIES
from train import macro_f1


TARGET_DELTA = np.array([2.544, 1.504, 1.435, 0.887])
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "simulation_output.txt"


def simulate(n, rng, gamma=1.0):
    y = rng.integers(0, 3, n)
    z = TARGET_DELTA[None, :, None] * np.eye(3)[y][:, None, :] + rng.normal(size=(n, 4, 3))
    lg = gamma * TARGET_DELTA[None, :, None] * z
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return y, e / e.sum(-1, keepdims=True)


def _format_report(res, seeds, n_test, pred, sup, S, fb, demo_p, demo_r, explanation):
    lines = [
        f"Emotion Fusion Simulation Output ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        f"Config: seeds={seeds}, n_test={n_test}, target_delta={TARGET_DELTA.tolist()}",
        "",
        f"Accuracy % (mean over {seeds} seeds, calibrated simulated posteriors)",
    ]
    summary_lines = [f"Accuracy % (mean over {seeds} seeds, calibrated simulated posteriors)"]
    for sc, d in res.items():
        row = f"{sc:12s} " + "  ".join(f"{k}: {np.mean(v):.2f}" for k, v in d.items())
        summary_lines.append(row)
        lines.append(
            f"{sc:12s} "
            + "  ".join(f"{k}: {np.mean(v):.2f} (std {np.std(v):.2f})" for k, v in d.items())
        )

    lines.extend(["", "Per-seed Accuracy %:"])
    for sc, d in res.items():
        lines.append(f"  [{sc}]")
        for k, v in d.items():
            vals = ", ".join(f"{x:.2f}" for x in v)
            lines.append(f"    {k:10s}: [{vals}]")

    demo_header = (
        f"\nDemo sample -> {CLASSES[pred[0]]} "
        f"support {sup[0].round(3)} S_c {S[0].round(3)}"
    )
    summary_lines.append(demo_header)
    lines.extend(["", "Demo sample inputs (reliability=" + str(demo_r.tolist()) + "):"])
    lines.append(f"  {'Modality':10s} " + "  ".join(f"{c:>8s}" for c in CLASSES))
    for mod, probs in zip(MODALITIES, demo_p):
        lines.append(f"  {mod:10s} " + "  ".join(f"{val:8.3f}" for val in probs))
    lines.append(demo_header.lstrip("\n"))
    lines.append(f"  fallback={bool(fb[0])}")
    for c, rules in explanation.items():
        rule_line = f"  {c:9s} {rules or 'none'}"
        summary_lines.append(rule_line)
        lines.append(rule_line)

    return "\n".join(summary_lines), "\n".join(lines) + "\n"


def run_simulation(seeds=5, n_test=3000, output_path=DEFAULT_OUTPUT_PATH):
    res = {}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        yv, Pv = simulate(2000, rng)
        r = reliability_from_validation([macro_f1(yv, Pv[:, m].argmax(1)) for m in range(4)])
        fz = MamdaniFusion(r)
        y, P = simulate(n_test, rng)
        full = np.ones((n_test, 4), bool)
        noeeg = full.copy()
        noeeg[:, 0] = False
        miss = rng.random((n_test, 4)) > 0.3
        miss[~miss.any(1), rng.integers(0, 4)] = True
        for name, A in [("All four", full), ("30% missing", miss), ("EEG missing", noeeg)]:
            preds = {"Mean": (P * A[..., None]).sum(1).argmax(1),
                     "W.mean": (P * (A * r)[..., None]).sum(1).argmax(1),
                     "DS undisc": ds_fuse(P, A)[0].argmax(1),
                     "DS disc": ds_fuse(P, A, r)[0].argmax(1),
                     "Mamdani": fz.predict(P, A)[0]}
            for k, p in preds.items():
                res.setdefault(name, {}).setdefault(k, []).append((p == y).mean() * 100)

    p = np.array([[.053, .003, .944], [.005, .009, .986], [.021, .328, .650], [.529, .172, .300]])
    demo_r = np.array([1.0, .739, .689, .445])
    fz = MamdaniFusion(demo_r)
    pred, sup, S, fb = fz.predict(p[None], np.ones((1, 4), bool))
    explanation = fz.explain(p, np.ones(4, bool))

    console_text, file_text = _format_report(
        res, seeds, n_test, pred, sup, S, fb, p, demo_r, explanation
    )
    print(console_text)

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(file_text, encoding="utf-8")
        print(f"\nSaved full output to: {out_file}")


if __name__ == "__main__":
    run_simulation()
