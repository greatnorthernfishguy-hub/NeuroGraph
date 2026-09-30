# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5) — turn 2: also writes recall_scenarios (cc_assemble_recall,
#   Pith-ON and gate-off, short items) from BASE.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 golden generator
# What: regenerates tests/fixtures/pith_clip_813_golden_base.json from BASE e4ebf982.
# Why: the golden must come from the OLD code, never from the branch under test.
# How: `git show e4ebf982:cc_ng_organism.py` -> temp file -> imported as a separate
#   module; run with `env -u NG_EMBED_REMOTE python tests/fixtures/gen_pith_clip_813_golden.py`.
# -------------------
import importlib.util, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
BASE = "e4ebf982b1989fd9066d610b94853bc68bf70d37"

src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:cc_ng_organism.py"])
tmp = tempfile.mkdtemp(prefix="cc_ng_organism_base_")
path = os.path.join(tmp, "cc_ng_organism_base.py")
open(path, "wb").write(src)
spec = importlib.util.spec_from_file_location("cc_ng_organism_base", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["cc_ng_organism_base"] = mod
spec.loader.exec_module(mod)

from pith_clip_813_scenarios import build_scenarios, build_recall_scenarios
golden = {"base_commit": BASE, "scenarios": build_scenarios(mod),
          "recall_scenarios": build_recall_scenarios(mod)}
out = os.path.join(ROOT, "tests", "fixtures", "pith_clip_813_golden_base.json")
with open(out, "w") as f:
    json.dump(golden, f, indent=1, sort_keys=True, ensure_ascii=False)
print("wrote", out, {k: (v["state"], v["assemblies"], len(v["context"])) for k, v in golden["scenarios"].items()},
      {k: len(v) for k, v in golden["recall_scenarios"].items()})
