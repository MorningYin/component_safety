from __future__ import annotations
from typing import List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path

class ModelConfig(BaseModel):
    alias: str
    path: str
    device: str = "cuda"
    dtype: str = "float16"

class DataConfig(BaseModel):
    benign_path: Path
    harmful_path: Path
    refusal_train_path: Optional[Path] = None
    benign_train_path: Optional[Path] = None
    batch_size: int = 32
    filter_train: bool = True
    filter_val: bool = True

class DirectionConfig(BaseModel):
    method: Literal["diff", "pca", "svm"] = "diff"
    layer_idx: Optional[int] = None  # If None, might use global or search


class SearchConfig(BaseModel):
    p_min: float = 0.05
    p_max: float = 0.2    # Percentile fraction (0.05 to 0.4 means 5th to 40th percentile)
    n_p: int = 20
    m_min: float = 0.1   # Percentage fraction (0.1 to 0.5 means 10% to 50% of subset)
    m_max: float = 0.3
    n_m: int = 10
    f1_tolerance: float = 0.001

class EvaluationConfig(BaseModel):
    num_autodan_goals: int = 100

class ExperimentConfig(BaseModel):
    experiment_id: str
    runs_root: Path = Path("results")
    model: ModelConfig
    data: DataConfig
    seed: int = 42
    direction: DirectionConfig = Field(default_factory=DirectionConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.experiment_id

    def ensure_dirs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
