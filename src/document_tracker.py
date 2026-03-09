import json
import hashlib
from pathlib import Path


class DocumentTracker:
    def __init__(self, tracker_path: str):
        self.tracker_path = Path(tracker_path)
        self._data: dict = {"files": {}, "embedding_mode": None}
        if self.tracker_path.exists():
            with open(self.tracker_path, "r") as f:
                self._data = json.load(f)

    @property
    def embedding_mode(self) -> str | None:
        return self._data.get("embedding_mode")

    def _file_hash(self, path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_changed(self, path: Path) -> bool:
        key = str(path)
        if key not in self._data["files"]:
            return True
        stored = self._data["files"][key]
        current_hash = self._file_hash(path)
        return stored.get("hash") != current_hash

    def update(self, path: Path, num_chunks: int):
        key = str(path)
        self._data["files"][key] = {
            "hash": self._file_hash(path),
            "chunks": num_chunks,
        }

    def clean_missing_files(self, existing_files: set) -> int:
        existing_keys = {str(p) for p in existing_files}
        missing = [k for k in self._data["files"] if k not in existing_keys]
        for k in missing:
            del self._data["files"][k]
        return len(missing)

    def get_stats(self) -> dict:
        files = self._data["files"]
        total_chunks = sum(v.get("chunks", 0) for v in files.values())
        return {"total_files": len(files), "total_chunks": total_chunks}

    def check_embedding_mode_compatibility(self, mode: str) -> bool:
        stored = self._data.get("embedding_mode")
        return stored is None or stored == mode

    def set_embedding_mode(self, mode: str):
        self._data["embedding_mode"] = mode

    def reset(self):
        self._data = {"files": {}, "embedding_mode": None}

    def save(self):
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_path, "w") as f:
            json.dump(self._data, f, indent=2)
