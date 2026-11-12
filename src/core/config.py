"""
Core application configuration and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Any
import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --------------------------------------------------------------------------
    # 1. Local Model Hub & ONNX Artifact Paths
    # --------------------------------------------------------------------------
    HUB_DIR: str = r"C:\Hub\Models\kahneman_guard"
    USE_LOCAL_ONNX: bool = False

    # Tier 1: ModernBERT Dense Engine
    DENSE_ONNX_PATH: str | None = None
    DENSE_TOKENIZER_ID: str | None = None
    DENSE_TEMPERATURE: float = 1.9665

    # Tier 1: ModernBERT Discrete Diffusion Engine (djev)
    DIFFUSION_ONNX_PATH: str | None = None
    DIFFUSION_TOKENIZER_ID: str | None = None
    DIFFUSION_TEMPERATURE: float = 1.3358

    # Tier 1: ModernBERT Diffusion + RLCD Engine
    DIFFUSION_RLCD_ONNX_PATH: str | None = None
    DIFFUSION_RLCD_TOKENIZER_ID: str | None = None
    DIFFUSION_RLCD_TEMPERATURE: float = 1.9460

    # Tier 2: System-Two Deep Loop (Llama-Guard-3-1B GGUF)
    TIER2_GGUF_PATH: str | None = None

    # --------------------------------------------------------------------------
    # 2. Server & Proxy Gateway Settings
    # --------------------------------------------------------------------------
    PROXY_HOST: str = "0.0.0.0"
    PROXY_PORT: int = 5950
    LOG_LEVEL: str = "info"

    # --------------------------------------------------------------------------
    # 3. Optional Microservice Backends
    # --------------------------------------------------------------------------
    TIER1_URL: str = "http://localhost:5816"
    TIER1_ENDPOINT: str = "/v1/system_one"
    TIER1_TIMEOUT_MS: int = 5000
    TIER1_MOCK_MODE: bool = False

    TIER2_URL: str = "http://localhost:5817"
    TIER2_ENDPOINT: str = "/v1/chat/completions"
    TIER2_TIMEOUT_MS: int = 15000
    TIER2_MOCK_MODE: bool = False
    TIER2_MODEL_NAME: str = "Llama-Guard-3-1B"

    # Upstream Target LLM (for forwarding safe requests)
    UPSTREAM_LLM_URL: str = "http://localhost:5814"
    UPSTREAM_LLM_MOCK: bool = True

    # --------------------------------------------------------------------------
    # 4. Router Confidence Thresholds & Modes
    # --------------------------------------------------------------------------
    ROUTING_CONFIG_PATH: str = "configs/routing_config.yaml"
    DEFAULT_ROUTING_MODE: str = "dual_process"
    DEFAULT_VARIANT: str = "dense"
    ROUTER_TAU_SAFE: float = 0.80
    ROUTER_TAU_UNSAFE: float = 0.80

    # --------------------------------------------------------------------------
    # 5. Evaluation & Benchmark Defaults
    # --------------------------------------------------------------------------
    BENCHMARK_BATCH_SIZE: int = 20
    BENCHMARK_RANDOM_SEED: int = 42
    BENCHMARK_CACHE_DIR: str = "data/benchmarks"
    DATASET_JAILBREAKBENCH: str = "data/benchmarks/jailbreakbench.jsonl"
    DATASET_TOXICCHAT: str = "data/benchmarks/toxicchat_test.jsonl"
    DATASET_BEAVERTAILS: str = "data/benchmarks/beavertails_test.jsonl"
    DATASET_XSTEST: str = "data/benchmarks/xstest_benign.jsonl"

    @model_validator(mode="after")
    def populate_hub_derived_paths(self):
        hub = Path(self.HUB_DIR)
        if not self.DENSE_ONNX_PATH:
            self.DENSE_ONNX_PATH = str(hub / "dense_guard" / "kahnemanguard_modernbert_int8.onnx")
        if not self.DENSE_TOKENIZER_ID:
            self.DENSE_TOKENIZER_ID = str(hub / "dense_guard")
        if not self.DIFFUSION_ONNX_PATH:
            self.DIFFUSION_ONNX_PATH = str(hub / "diff_guard" / "kahnemanguard_diffusion_int8.onnx")
        if not self.DIFFUSION_TOKENIZER_ID:
            self.DIFFUSION_TOKENIZER_ID = str(hub / "diff_guard")
        if not self.DIFFUSION_RLCD_ONNX_PATH:
            self.DIFFUSION_RLCD_ONNX_PATH = str(hub / "diff_rlcd_guard" / "kahnemanguard_diffusion_rlcd_int8.onnx")
        if not self.DIFFUSION_RLCD_TOKENIZER_ID:
            self.DIFFUSION_RLCD_TOKENIZER_ID = str(hub / "diff_rlcd_guard")
        if not self.TIER2_GGUF_PATH:
            self.TIER2_GGUF_PATH = str(hub.parent / "Llama-Guard-3-1B.Q4_K_M.gguf")
        return self


settings = Settings()


def load_routing_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads routing_config.yaml and resolves ${VAR} placeholders using Settings."""
    target_path = Path(config_path or settings.ROUTING_CONFIG_PATH)
    if not target_path.exists():
        base_dir = Path(__file__).resolve().parent.parent.parent
        target_path = base_dir / "configs" / "routing_config.yaml"

    if not target_path.exists():
        return {}

    with open(target_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        val = getattr(settings, var_name, None)
        if val is None:
            val = os.getenv(var_name, "")
        # Normalize backslashes to forward slashes to prevent YAML escape sequence errors
        return str(val).replace("\\", "/")

    expanded = re.sub(r"\$\{([A-Za-z0-9_]+)\}", replacer, raw_text)
    return yaml.safe_load(expanded)
