from pathlib import Path
from typing import Optional
import yaml
from components_safety.core.config_schema import ExperimentConfig
from components_safety.core.manifest import Manifest
from components_safety.core.cache import GLOBAL_CACHE

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.ensure_dirs()
        self.manifest = Manifest(self.config.run_dir)
        self.cache = GLOBAL_CACHE
        
        # Save the config itself for reproducibility
        self.save_config()

    def save_config(self):
        config_path = self.config.run_dir / "config.yaml"
        # Using dict() to serialize for yaml
        with open(config_path, "w") as f:
            yaml.dump(self.config.dict(), f, sort_keys=False)
        self.manifest.log_output("run_config", config_path)

    def run_stage(self, name: str, func, *args, **kwargs):
        print(f"\n>>> Running Stage: {name}")
        self.manifest.log_stage(name, "started")
        try:
            start_time = time.time()
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            self.manifest.log_stage(name, "completed", {"duration": duration})
            return result
        except Exception as e:
            self.manifest.log_stage(name, "failed", {"error": str(e)})
            raise e

import time # For time.time() in run_stage
