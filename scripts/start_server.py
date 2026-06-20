from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT_DIR / "logs"
STATE_FILE = RUNTIME_DIR / "managed_server.json"
STOP_FILE = RUNTIME_DIR / "managed_server.stop.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start one managed SmartRecipe development server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=300,
        help="Stop after this many seconds without an HTTP request; use 0 to disable (default: 300).",
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable Uvicorn auto-reload.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stop_previous_managed_server() -> None:
    previous = read_json(STATE_FILE)
    token = str(previous.get("token", ""))
    if not token:
        return

    print(f"检测到上一份 SmartRecipe 服务，正在关闭（端口 {previous.get('port', '?')}）...")
    write_json(STOP_FILE, {"token": token})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = read_json(STATE_FILE)
        if current.get("token") != token:
            return
        time.sleep(0.2)

    # The old launcher did not answer its stop request. Only terminate the PID
    # recorded in this project's own state file.
    old_pid = previous.get("launcher_pid")
    if isinstance(old_pid, int) and old_pid > 0:
        terminate_process_tree(old_pid)
        time.sleep(0.5)


def terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def stop_child_process(process: subprocess.Popen) -> None:
    """Stop Uvicorn and fall back to its direct process if tree cleanup fails."""
    if process.poll() is not None:
        return
    terminate_process_tree(process.pid)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.terminate()
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def cleanup(token: str, activity_file: Path) -> None:
    if read_json(STATE_FILE).get("token") == token:
        STATE_FILE.unlink(missing_ok=True)
    if read_json(STOP_FILE).get("token") == token:
        STOP_FILE.unlink(missing_ok=True)
    activity_file.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port 必须在 0 到 65535 之间")
    if args.idle_timeout < 0:
        raise SystemExit("--idle-timeout 不能为负数")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    stop_previous_managed_server()

    token = uuid.uuid4().hex
    activity_file = RUNTIME_DIR / f"server_activity_{token}.heartbeat"
    activity_file.touch()
    env = os.environ.copy()
    env["SMART_RECIPE_ACTIVITY_FILE"] = str(activity_file)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        command.append("--reload")

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=ROOT_DIR, env=env, creationflags=creationflags)
    write_json(
        STATE_FILE,
        {
            "token": token,
            "launcher_pid": os.getpid(),
            "server_pid": process.pid,
            "host": args.host,
            "port": args.port,
            "idle_timeout": args.idle_timeout,
        },
    )
    print(f"SmartRecipe 服务已启动：http://{args.host}:{args.port}/ui")
    if args.idle_timeout:
        print(f"连续 {args.idle_timeout} 秒没有 HTTP 请求时将自动关闭。")

    try:
        while process.poll() is None:
            if read_json(STOP_FILE).get("token") == token:
                print("收到新启动器的关闭请求，正在停止当前服务...")
                break
            if args.idle_timeout:
                try:
                    idle_seconds = time.time() - activity_file.stat().st_mtime
                except OSError:
                    idle_seconds = 0
                if idle_seconds >= args.idle_timeout:
                    print(f"连续 {args.idle_timeout} 秒没有 HTTP 请求，正在自动关闭服务...")
                    break
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在关闭 SmartRecipe 服务...")
    finally:
        stop_child_process(process)
        cleanup(token, activity_file)
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
