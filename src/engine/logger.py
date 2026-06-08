import json
import os
import time
from datetime import datetime
from typing import Any, Dict

class Logger:

    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.jsonl_path = os.path.join(save_dir, "metrics.jsonl")
        self.text_path = os.path.join(save_dir, "train.log")
        self._jsonl = open(self.jsonl_path, "a", buffering=1)
        self._text = open(self.text_path, "a", buffering=1)
        self._start = time.time()

    def close(self) -> None:
        for fp in (self._jsonl, self._text):
            try:
                fp.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def log(self, metrics: Dict[str, Any], step: int, phase: str = "train") -> None:
        record = {"step": step, "phase": phase, "elapsed": round(time.time() - self._start, 2)}
        for k, v in metrics.items():
            try:
                record[k] = float(v)
            except (TypeError, ValueError):
                record[k] = v
        self._jsonl.write(json.dumps(record) + "\n")

    def info(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self._text.write(line + "\n")
        try:
            from tqdm import tqdm
            tqdm.write(line)
        except Exception:
            print(line, flush=True)
