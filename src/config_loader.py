# config_loader.py
import os, json, argparse
from pathlib import Path
from typing import Any
from dotenv import dotenv_values
import yaml

from config_schema import AppConfig, LLMConfig, GenPlanStage, BuildKGStage, UseKGStage, DesignPaths

def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_app_config(
    designs_yaml: str = "designs.yaml",
    overrides_json: str = "",
    env_prefix: str = "AF_"  # AssertionForge
) -> AppConfig:
    if not Path(designs_yaml).exists():
        raise FileNotFoundError(f"Config file {designs_yaml} not found")

    data = yaml.safe_load(Path(designs_yaml).read_text()) or {}

    # Build AppConfig from YAML structure
    cfg = AppConfig(
        task=data.get("task", "gen_plan"),
        design_name=data.get("design_name", "uart"),
        llm=LLMConfig(**data.get("llm", {})),
        gen_plan=GenPlanStage(**data.get("gen_plan", {})),
        build_KG=BuildKGStage(**data.get("build_KG", {})),
        use_KG=UseKGStage(**data.get("use_KG", {})),
        designs={name: DesignPaths(**entry) for name, entry in data.get("designs", {}).items()},
    )

    # 2) Apply ENV overrides (flat)
    for k, v in os.environ.items():
        if k.startswith(env_prefix):
            key = k[len(env_prefix):]
            if key == "task": cfg.task = v
            elif key == "design_name": cfg.design_name = v
            elif key == "llm_model": cfg.llm.model = v
            elif key == "use_KG": cfg.gen_plan.use_KG = v.lower() == "true"

    # 3) Apply structured JSON overrides (path-like keys)
    if overrides_json:
        try:
            obj = json.loads(overrides_json)
            merged = _deep_update(cfg.__dict__, obj)
            cfg = AppConfig(**merged)
        except Exception as e:
            raise ValueError(f"Invalid overrides_json: {e}")
    
    # after building cfg
    dp = cfg.designs.get(cfg.design_name)
    if dp and dp.env_source_path and Path(dp.env_source_path).exists():
        env_vars = dotenv_values(dp.env_source_path)
        if "GRAPHRAG_API_KEY" in env_vars:
            cfg.llm.args["api_key"] = env_vars["GRAPHRAG_API_KEY"]

    return cfg

def build_FLAGS_from_cli() -> Any:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["gen_plan","build_KG","use_KG"])
    p.add_argument("--design_name")
    p.add_argument("--designs_yaml", default="designs.yaml")
    p.add_argument("--overrides_json", default="")
    p.add_argument("--valid_signals", nargs="+", help="List of architectural signals")
    args, _ = p.parse_known_args()

    cfg = load_app_config(designs_yaml=args.designs_yaml, overrides_json=args.overrides_json)
    if args.task: cfg.task = args.task
    if args.design_name: cfg.design_name = args.design_name
    if args.valid_signals: cfg.gen_plan.valid_signals = args.valid_signals


    return cfg.to_FLAGS()
