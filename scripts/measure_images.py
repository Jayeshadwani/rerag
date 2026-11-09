"""Measures a built image against Render's free limits (proposal 009): size, start, peak memory.

Usage: `python scripts/measure_images.py NAME IMAGE HEALTH_PATH CONTAINER_PORT [--post-ask]
[-e KEY=VALUE ...]`. It starts the image with a 512 MB memory limit, waits for the health URL
(cold start = container start to the first 200), optionally sends a few questions (the API's
retrieval and ONNX embedding run before the model call, which fails offline with a fake key),
reads the container's peak memory from its cgroup and appends the row to
`results/deploy_measurements.json`. Numbers are measured on this machine, not on Render.
"""

import argparse
import contextlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "deploy_measurements.json"
HOST_PORT = 18080
LIMIT_MB = 512


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def peak_bytes(name: str) -> int:
    """The container's peak memory: cgroup v2 `memory.peak`, else v1 `max_usage_in_bytes`."""
    script = (
        "cat /sys/fs/cgroup/memory.peak 2>/dev/null || "
        "cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes"
    )
    return int(sh("docker", "exec", name, "sh", "-c", script))


def wait_health(name: str, path: str, timeout_s: float) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if sh("docker", "inspect", name, "--format", "{{.State.Status}}") != "running":
            return None  # the container died (for example, out of memory)
        try:
            with urllib.request.urlopen(f"http://localhost:{HOST_PORT}{path}", timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.25)
    return None


def post_ask() -> None:
    for question in ("what is linear regression?", "how do I split data into train and test?"):
        body = json.dumps({"question": question, "history": []}).encode()
        req = urllib.request.Request(
            f"http://localhost:{HOST_PORT}/ask",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        # offline the model call fails after retrieval, which is the load we want to measure
        with contextlib.suppress(urllib.error.HTTPError):
            urllib.request.urlopen(req, timeout=60)  # noqa: S310


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("image")
    parser.add_argument("health")
    parser.add_argument("port")
    parser.add_argument("--post-ask", action="store_true")
    parser.add_argument("--memory-mb", type=int, default=LIMIT_MB)
    parser.add_argument("-e", action="append", default=[])
    args = parser.parse_args()
    name = f"measure-{args.name}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)  # noqa: S603, S607
    env = [x for pair in args.e for x in ("-e", pair)]
    sh("docker", "run", "-d", "--name", name, "--memory", f"{args.memory_mb}m", "--memory-swap",
       f"{args.memory_mb}m", "-p", f"{HOST_PORT}:{args.port}", *env, args.image)  # fmt: skip
    try:
        cold = wait_health(name, args.health, 300)
        if cold is not None and args.post_ask:
            post_ask()
        time.sleep(2)
        state = json.loads(sh("docker", "inspect", name))[0]["State"]
        peak = None if state["Status"] != "running" else peak_bytes(name)
    finally:
        sh("docker", "rm", "-f", name)
    size = int(sh("docker", "image", "inspect", args.image, "--format", "{{.Size}}"))
    row = {
        "image_size_mb": round(size / 1e6),
        "cold_start_s": None if cold is None else round(cold, 1),
        "peak_memory_mb": None if peak is None else round(peak / 2**20),
        "memory_limit_mb": args.memory_mb,
        "oom_killed": bool(state["OOMKilled"]),
        "asked_questions": args.post_ask,
    }
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[args.name] = row
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(args.name, row)


if __name__ == "__main__":
    main()
