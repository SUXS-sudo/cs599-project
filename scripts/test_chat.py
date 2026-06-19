from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    default_message = "\u6211\u5bb6\u91cc\u6709\u9e21\u86cb\u548c\u756a\u8304\uff0c\u53ef\u4ee5\u505a\u4ec0\u4e48\uff1f"
    message = " ".join(sys.argv[1:]).strip() or default_message
    data = json.dumps(
        {"message": message, "session_id": "demo-user", "top_k": 3},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8010/chat",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    timeout = int(os.getenv("CHAT_TEST_TIMEOUT", "120"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}")
        return 1
    except TimeoutError:
        print(f"Request timed out after {timeout} seconds. The server may still be waiting for the LLM response.")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
