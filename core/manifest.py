import json
import time
import socket
import getpass
import platform
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union

class Manifest:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "manifest.json"
        self.data = {
            "experiment_start_time": datetime.now().isoformat(),
            "system_info": {
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "os": platform.platform(),
            },
            "stages": {},
            "outputs": {}
        }

    def log_stage(self, stage_name: str, status: str, metadata: Dict[str, Any] = None):
        self.data["stages"][stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "metadata": metadata or {}
        }
        self.save()

    def log_output(self, key: str, path: Union[str, Path]):
        self.data["outputs"][key] = str(path)
        self.save()

    def _stringify_keys(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._stringify_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._stringify_keys(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        return obj

    def save(self):
        clean_data = self._stringify_keys(self.data)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
