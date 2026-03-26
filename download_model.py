import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO = os.environ["MODEL_REPO"]
MODEL_PATH = Path(os.environ["MODEL_PATH"])
HF_HOME = Path(os.environ.get("HF_HOME", "/tmp/huggingface"))
MODELS_DIR = MODEL_PATH.parent


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODELS_DIR),
        allow_patterns=["*.q8_0.gguf", "*.Q8_0.gguf"],
        ignore_patterns=["*mmproj*"],
    )

    matches = [
        path
        for path in MODELS_DIR.rglob("*.gguf")
        if "mmproj" not in path.name.lower()
        and "q8_0" in path.name.lower()
        and path.is_file()
    ]

    if not matches:
        raise RuntimeError(f"No Q8_0 GGUF file found in {MODEL_REPO}")

    source = matches[0]
    if source != MODEL_PATH:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        shutil.move(str(source), str(MODEL_PATH))

    for path in list(MODELS_DIR.iterdir()):
        if path == MODEL_PATH:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if HF_HOME.exists():
        shutil.rmtree(HF_HOME)


if __name__ == "__main__":
    main()
