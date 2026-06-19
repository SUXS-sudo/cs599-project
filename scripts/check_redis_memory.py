from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Redis memory and preference storage.")
    parser.add_argument("--session-id", default="redis-smoke", help="Session id used for smoke test.")
    args = parser.parse_args()

    from app.services.redis_memory import RedisMemoryStore

    try:
        store = RedisMemoryStore(max_messages=10)
        store.add_turn(args.session_id, "我不吃牛肉，对虾过敏，喜欢清淡。", "已记录你的偏好。")
        prefs = store.update_preferences(
            args.session_id,
            preferences=["清淡"],
            allergies=["虾"],
            dislikes=["牛肉"],
        )
        history = store.get_history(args.session_id)
    except Exception as exc:
        print(f"redis_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"redis_backend={store.backend}")
    print(f"redis_url={store.redis_url}")
    print(f"session_id={args.session_id}")
    print(f"history_messages={len(history)}")
    print(f"preferences={','.join(prefs.preferences)}")
    print(f"allergies={','.join(prefs.allergies)}")
    print(f"dislikes={','.join(prefs.dislikes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
