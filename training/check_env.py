from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.train_lora import REQUIRED_STACK
else:
    from training.train_lora import REQUIRED_STACK


OPTIONAL_STACK = ["trl", "sentencepiece", "safetensors"]


def detect_modules(module_names: list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for module_name in module_names:
        try:
            __import__(module_name)
            present.append(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return present, missing


def build_env_report() -> dict:
    required_present, required_missing = detect_modules(REQUIRED_STACK)
    optional_present, optional_missing = detect_modules(OPTIONAL_STACK)
    return {
        "required_stack": REQUIRED_STACK,
        "required_present": required_present,
        "required_missing": required_missing,
        "optional_stack": OPTIONAL_STACK,
        "optional_present": optional_present,
        "optional_missing": optional_missing,
        "ready_for_local_lora_smoke": not required_missing,
    }


def main() -> None:
    print(json.dumps(build_env_report(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
