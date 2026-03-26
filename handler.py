import atexit
import json
import os
import subprocess
import threading
import time
from collections import deque
from typing import Any

import requests
import runpod


class LlamaServer:
    def __init__(self) -> None:
        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.model_path = os.getenv("MODEL_PATH", "/models/model.gguf")
        self.model_name = os.getenv(
            "RUNPOD_MODEL_NAME",
            "qwen3.5-4b-claude-opus-reasoning-q8_0",
        )
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._logs: deque[str] = deque(maxlen=200)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ensure_started(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                return

            command = [
                os.getenv("LLAMA_SERVER_BIN", "/app/llama-server"),
                "-m",
                self.model_path,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "-c",
                os.getenv("LLAMA_CTX_SIZE", "8192"),
                "-np",
                os.getenv("LLAMA_PARALLEL", "1"),
                "-ngl",
                os.getenv("LLAMA_GPU_LAYERS", "999"),
                "-b",
                os.getenv("LLAMA_BATCH", "1024"),
            ]

            if os.getenv("LLAMA_JINJA", "1") == "1":
                command.append("--jinja")

            print(f"Starting llama-server: {' '.join(command)}", flush=True)
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._drain_logs, daemon=True).start()
            self._wait_for_ready()

    def _drain_logs(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip()
            self._logs.append(line)
            print(f"[llama-server] {line}", flush=True)

    def _wait_for_ready(self) -> None:
        deadline = time.time() + 600
        while time.time() < deadline:
            if not self.process or self.process.poll() is not None:
                break

            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if response.ok:
                    return
            except requests.RequestException:
                pass

            time.sleep(2)

        recent_logs = "\n".join(self._logs) or "no logs captured"
        raise RuntimeError(f"llama-server failed to start.\n{recent_logs}")

    def stop(self) -> None:
        with self._lock:
            if not self.process:
                return
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None


server = LlamaServer()
atexit.register(server.stop)


def _normalize_input(job_input: Any) -> dict[str, Any]:
    if isinstance(job_input, str):
        return {"prompt": job_input}

    if not isinstance(job_input, dict):
        raise ValueError("job input must be a JSON object or a raw prompt string")

    payload = dict(job_input)

    if payload.get("healthcheck"):
        return {"healthcheck": True}

    if "messages" not in payload and "prompt" not in payload:
        prompt = payload.pop("input", None)
        if isinstance(prompt, str):
            payload["prompt"] = prompt

    if "max_new_tokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("max_new_tokens")

    payload.setdefault("stream", False)
    payload.setdefault("model", server.model_name)

    return payload


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{server.base_url}{path}",
        json=payload,
        timeout=(30, 1800),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"llama-server request failed with {response.status_code}: {response.text}"
        ) from exc
    return response.json()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    server.ensure_started()
    payload = _normalize_input(job.get("input"))

    if payload.get("healthcheck"):
        return {
            "status": "ok",
            "model": server.model_name,
            "model_path": server.model_path,
            "base_url": server.base_url,
        }

    if "messages" in payload:
        return _post_json("/v1/chat/completions", payload)

    if "prompt" in payload:
        return _post_json("/v1/completions", payload)

    raise ValueError("job input must include either `messages` or `prompt`")


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "status": "booting",
                "model": server.model_name,
                "model_path": server.model_path,
                "base_url": server.base_url,
            }
        ),
        flush=True,
    )
    runpod.serverless.start({"handler": handler})
