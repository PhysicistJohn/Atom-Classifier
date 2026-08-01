"""Summarize G2 result JSONs: pace/memory/accuracy/skips matrix + torch."""
import json
import sys
from pathlib import Path

SP = Path("/private/tmp/claude-501/-Users-johnelliott-PersonalGitHub/"
          "6179ba93-fb7a-4112-b70b-19540b0f4fce/scratchpad")
DWELLS = ("1ms", "2.5ms", "10ms")


def final_eval(res):
    f = res["final"]
    return {w: (f[w]["balanced_accuracy"], f[w]["min_profile_recall"])
            for w in DWELLS}, f["eval_rows_used"], f["escalation"]


def show(tag, path, torch_final=None):
    res = json.loads(Path(path).read_text())
    fe, rows, esc = final_eval(res)
    print(f"== {tag}  wall={res['wall_s']:.1f}s  rows={rows}")
    for w in DWELLS:
        line = (f"  {w:5s} bal={fe[w][0]:.4f} minCell={fe[w][1]:.3f}")
        if torch_final:
            line += f"  d_bal={fe[w][0]-torch_final[w][0]:+.4f}"
        print(line)
    print(f"  esc@1ms answered={esc['answered_frac']:.3f} "
          f"acc={esc['answered_accuracy']:.3f}")
    d = res.get("mlx_diagnostics")
    if d:
        print(f"  skipped_steps={res['skipped_steps']}  "
              f"peak={d['peak_memory_bytes']/1e9:.2f}GB  "
              f"active_end={d['active_memory_bytes_end']/1e9:.2f}GB  "
              f"boot={d['boot_s']:.0f}s")
        prec = d["precision"]
        print(f"  dtype={prec['dtype']} tf32_active={prec['tf32_active']} "
              f"relerr={prec['matmul_rel_err_vs_fp64']:.2e}")
        for k, v in d["pace_ms_per_ep"].items():
            print(f"  pace {k:14s} n={v['count']:3d} mean={v['mean']:7.0f} "
                  f"min={v['min']:7.0f} max={v['max']:7.0f} ms")
    return fe


if __name__ == "__main__":
    tfe = None
    if (SP / "g2_torch300.json").exists():
        tfe = show("torch", SP / "g2_torch300.json")
    for tag in sys.argv[1:]:
        show(tag, SP / f"g2_mlx_{tag}.json", tfe)
