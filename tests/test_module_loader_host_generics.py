# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1) — module loader keeps the host's complete modules
# What: runs the real _bootstrap_modules() (extracted by AST from neurograph_rpc.py) in a subprocess against a
#       fake registry: a module that vendors its own ng_embed, a module that vendors none, a loader-window
#       intrusion that plants a half-built ng_lite, and a thread hammering "from ng_embed import embed".
# Why:  VPS 2026-10-24 21:32Z: sys.modules['ng_embed'] was left as a module with __file__ but no
#       NGEmbed/embed/embed_batch (hash fallback for CC's VPS mind, Syl's dual-pass and lazy expansion
#       failing). Cause: the loader popped a module that a concurrent thread was still executing, and
#       restored "whatever was there" instead of the host's own object.
# How:  subprocess (sys.modules is process-global); HOME points at a temp registry; assertions printed as JSON.
# -------------------
import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''
import ast, json, logging, os, sys, threading, time
ROOT, HOST = sys.argv[1], sys.argv[2]
sys.path.insert(0, HOST)
import ng_embed as host_embed
import ng_lite as host_lite
src = open(os.path.join(ROOT, "neurograph_rpc.py")).read()
tree = ast.parse(src)
want = {"_bootstrap_modules", "_is_generic_vendored_name", "_GENERIC_PREFIXES", "_SHARED_HOST_MODULES"}
body = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in want:
        body.append(n)
    elif isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want for t in n.targets):
        body.append(n)
ns = {"os": os, "sys": sys, "time": time, "logger": logging.getLogger("t"), "List": list,
      "_module_instances": {}}
exec(compile(ast.Module(body=body, type_ignores=[]), "neurograph_rpc.py", "exec"), ns)

errors = []
stop = threading.Event()
def hammer():
    while not stop.is_set():
        try:
            from ng_embed import embed  # noqa: F401
        except Exception as e:  # record, keep going
            errors.append(repr(e))
        time.sleep(0.0005)
t = threading.Thread(target=hammer, daemon=True); t.start()
started = ns["_bootstrap_modules"]()
stop.set(); t.join()
m = sys.modules.get("ng_embed")
print(json.dumps({
    "started": started,
    "embed_is_host": m is host_embed,
    "embed_complete": all(hasattr(m, n) for n in ("NGEmbed", "embed", "embed_batch")),
    "lite_is_host": sys.modules.get("ng_lite") is host_lite,
    "a_saw_embed_file": os.environ.get("_A_EMBED_FILE_OUT") and open(os.environ["_A_EMBED_FILE_OUT"]).read(),
    "a_saw_lite_file": open(os.environ["_A_LITE_FILE_OUT"]).read(),
    "hammer_errors": sorted(set(errors))[:5],
}))
'''

HOOK = '''
import os, sys, time
class {cls}:
    MODULE_ID = "{mid}"
    def __init__(self):
        {body}
    def _module_on_message(self, *a, **k):
        pass
'''


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))


def test_loader_keeps_host_embed_and_restores_host_generics(tmp_path):
    host = tmp_path / "host"
    _write(host / "ng_embed.py", (ROOT / "ng_embed.py").read_text())
    _write(host / "ng_lite.py", "WHO = 'host'\n")
    a = tmp_path / "mods" / "a"
    _write(a / "ng_embed.py", "WHO = 'a-old-copy'\n")
    _write(a / "ng_lite.py", "WHO = 'a'\n")
    out_embed = tmp_path / "a_embed.txt"
    out_lite = tmp_path / "a_lite.txt"
    body_a = ("import ng_embed, ng_lite; "
              f"open({str(out_embed)!r}, 'w').write(ng_embed.__file__); "
              f"open({str(out_lite)!r}, 'w').write(ng_lite.WHO); time.sleep(0.05)")
    _write(a / "hook.py", HOOK.format(cls="AHook", mid="a", body=body_a))
    b = tmp_path / "mods" / "b"
    # b vendors nothing; while it loads, an intruder plants a half-built ng_lite under the host's name
    body_b = ("import types; m = types.ModuleType('ng_lite'); m.__file__ = 'half-built'; "
              "sys.modules['ng_lite'] = m; time.sleep(0.05)")
    _write(b / "hook.py", HOOK.format(cls="BHook", mid="b", body=body_b))
    reg = tmp_path / "home" / ".et_modules" / "registry.json"
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({"modules": {
        "a": {"install_path": str(a), "entry_point": "hook.py"},
        "b": {"install_path": str(b), "entry_point": "hook.py"},
    }}))
    env = dict(os.environ, HOME=str(tmp_path / "home"), _A_EMBED_FILE_OUT=str(out_embed),
               _A_LITE_FILE_OUT=str(out_lite))
    drv = tmp_path / "drv.py"
    drv.write_text(DRIVER)
    r = subprocess.run([sys.executable, str(drv), str(ROOT), str(host)], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["started"] == ["a", "b"]
    assert res["embed_is_host"] and res["embed_complete"]
    # the module's load-time import of ng_embed resolves to the host's one shared embedder module
    assert res["a_saw_embed_file"] == str(host / "ng_embed.py")
    # other generics stay isolated per module during its load ...
    assert res["a_saw_lite_file"] == "a"
    # ... and the host's own object comes back, even over a half-built intruder
    assert res["lite_is_host"]
    assert res["hammer_errors"] == []
