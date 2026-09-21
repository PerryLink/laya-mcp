"""Does the handshake answer immediately while the model is still loading?

The regression this guards is in both directions. Preloading before `server.run()`
makes the handshake take as long as the load - 19 s uncontended, 275 s while
another model holds the GPU - and both harnesses measured here time out at 30 s,
so the client never sees the tool list. Loading lazily *on the loop thread*
instead hangs the first `tools/call` forever, which is why the preload was there
in the first place.

This asserts the third option: handshake fast, tool call correct, and the call
arrives while the load is still in flight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

PY = r"D:\Projects\laya-family\.venv-laya\Scripts\python.exe"
CALL_TIMEOUT = 300.0


def send(proc: subprocess.Popen, payload: dict) -> None:
    proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    proc.stdin.flush()


def read_until(proc: subprocess.Popen, want_id: int, deadline: float) -> tuple[dict, float]:
    start = time.time()
    while time.time() - deadline < 0:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the server closed stdout")
        message = json.loads(line.decode("utf-8"))
        if message.get("id") == want_id:
            return message, time.time() - start
    raise TimeoutError(f"no reply to id={want_id} within the deadline")


def main() -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [PY, "-m", "laya_mcp", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        # 1. The handshake. This is the whole test.
        send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "latency-probe", "version": "0"},
            },
        })
        reply, elapsed = read_until(proc, 1, time.time() + 60)
        server = reply.get("result", {}).get("serverInfo", {})
        print(f"  initialize   {elapsed:6.2f}s   {server.get('name')} {server.get('version')}")
        handshake_ok = elapsed < 5.0

        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 2. The tool list, which also must not wait for the model.
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        reply, elapsed = read_until(proc, 2, time.time() + 60)
        tools = [t["name"] for t in reply.get("result", {}).get("tools", [])]
        print(f"  tools/list   {elapsed:6.2f}s   {len(tools)} tools: {', '.join(sorted(tools))}")
        list_ok = elapsed < 5.0 and "laya_noul" in tools

        # 3. A real call. It waits for the model, which is the point - and this is
        #    the call that used to hang forever when the load happened on the loop.
        send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "laya_noul",
                "arguments": {
                    "state": {"body": "The build fails on Windows because the launcher cannot find python."},
                    "instructions": "Is this a bug report?",
                    "boundary": {"true": "a defect in existing behaviour", "false": "a request for something new"},
                },
            },
        })
        reply, elapsed = read_until(proc, 3, time.time() + CALL_TIMEOUT)
        content = reply.get("result", {}).get("content") or []
        text = "\n".join(block.get("text", "") for block in content)
        payload = json.loads(text) if text.strip().startswith("{") else {}
        answer = (payload.get("answers") or {}).get("q") or {}
        print(f"  tools/call   {elapsed:6.2f}s   noul={answer.get('noul')} "
              f"band={answer.get('band')} model={payload.get('model')}")
        call_ok = answer.get("noul") is not None

        print()
        print(f"  handshake under 5s : {handshake_ok}")
        print(f"  tool list under 5s : {list_ok}")
        print(f"  call returned      : {call_ok}")
        return 0 if (handshake_ok and list_ok and call_ok) else 1
    finally:
        proc.kill()


if __name__ == "__main__":
    sys.exit(main())
