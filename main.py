"""Entry point: load a YAML config and dispatch to the named module/function.

    python main.py --config config/bongard_ow/ca_qwen25_14b.yaml

Env vars in YAML string values are expanded with ${VAR} syntax, so one config
runs unchanged locally and on HPC (set DATA_DIR / OUTPUT_DIR / DVRL_IMAGE_DIR in
.env or the environment).

This drives the `pri` package (src/pri). It is unrelated to code/main.py, which
belongs to the frozen NS-Reasoner provenance for the inherited COLM tables.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _load_config(path: str) -> dict:
    with open(path) as f:
        raw = os.path.expandvars(f.read())
    # expandvars leaves an unset ${VAR} verbatim, which would otherwise surface far
    # downstream as a missing file literally named '${DATA_DIR}/...'. Fail here instead.
    missing = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw)))
    if missing:
        raise SystemExit(
            f"{path}: unset environment variable(s): {', '.join(missing)}.\n"
            f"Set them in .env (copy .env.example) or export them before running."
        )
    return yaml.safe_load(raw)


def _dispatch(module_name: str, function_name: str, parameters: dict):
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load {function_name} from {module_name}: {e}") from e
    return func(**parameters)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a model script with a YAML config.")
    ap.add_argument("--config", required=True, help="Path to the YAML config file.")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override one config parameter, e.g. --set num_samples=5. Repeatable. "
             "VALUE is parsed as YAML, so 5 is an int and true is a bool. Intended for "
             "smoke tests, so the committed config keeps the production value.",
    )
    args = ap.parse_args()

    # Make src/ importable from a checkout without requiring `pip install -e`.
    src_dir = Path(__file__).resolve().parent / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    load_dotenv()

    cfg = _load_config(args.config)["model"]

    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in cfg["parameters"]:
            raise SystemExit(
                f"--set {key}: not a parameter of {args.config}. "
                f"Known: {sorted(cfg['parameters'])}"
            )
        cfg["parameters"][key] = yaml.safe_load(raw)
        print(f"override: {key} = {cfg['parameters'][key]!r}")

    print(f"Running {cfg['function']} from {cfg['module']} with parameters: {cfg['parameters']}")
    result = _dispatch(cfg["module"], cfg["function"], cfg["parameters"])
    print(f"Done: {result}")


if __name__ == "__main__":
    main()
