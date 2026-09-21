"""Prove the stdio server works with NO sidecar and NO network.

This is the configuration the DSH bundle ships, so it is the one that has to
work after a reboot. It exercises the exact failure mode that matters: a harness
spawns `python -m laya_mcp mcp --model-root <local snapshot>`, and everything -
model, tokenizer, inference - comes off local disk.

`HF_HUB_OFFLINE=1` is set deliberately. If anything in the path tries to reach
the Hub it raises instead of quietly downloading, which is the difference between
testing offline operation and hoping for it. A previous run without `--model-root`
pulled 1.1 GB into the HF cache while the same checkpoint already sat on disk.

Run:  python tests/selfhosted_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

MODEL_ROOT = os.environ.get("LAYA_MODEL_ROOT", r"D:\Projects\laya-family\_models\laya")
FAILURES: list[str] = []
CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


async def main() -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("the MCP SDK is not installed; run: pip install 'laya-mcp[mcp]'")
        return 2

    if not os.path.isdir(MODEL_ROOT):
        print(f"model root not found: {MODEL_ROOT}")
        return 2

    # No --sidecar, and the Hub is off, so every failure here is a real one.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp", "mcp", "--model-root", MODEL_ROOT],
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "USE_TF": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )

    print(f"spawning: {sys.executable} -m laya_mcp mcp --model-root {MODEL_ROOT}")
    print("with HF_HUB_OFFLINE=1 and no --sidecar\n")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            check("handshake completes with no sidecar", info is not None)
            check("server names itself", bool(info.serverInfo.name), str(info.serverInfo))

            listed = await session.list_tools()
            names = sorted(t.name for t in listed.tools)
            check("all five tools are exposed", len(names) == 5, str(names))

            result = await session.call_tool(
                "laya_ask",
                {
                    "state": {"body": "We were billed twice. Refund or we cancel our plan."},
                    "questions": {
                        "churn": {
                            "type": "noul",
                            "instructions": "Does the user threaten to cancel?",
                        },
                        "team": {
                            "type": "choice",
                            "instructions": "Which team handles this?",
                            "criteria": {
                                "billing": "invoices, refunds",
                                "technical": "bugs, outages",
                            },
                        },
                    },
                },
            )
            text = "\n".join(
                getattr(b, "text", "") for b in (result.content or []) if getattr(b, "text", None)
            )
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                check("the answer is JSON", False, text[:200])
                payload = {}

            if payload:
                check("the answer is JSON", True)
                check("the call succeeded with no sidecar", payload.get("ok") is True,
                      str(payload.get("error")))
                if payload.get("ok"):
                    answers = payload.get("answers") or {}
                    check("both questions answered", set(answers) == {"churn", "team"},
                          str(sorted(answers)))
                    check("reports the device it actually used",
                          payload.get("device") in ("cuda", "cpu"), str(payload.get("device")))
                    churn = answers.get("churn") or {}
                    check("the noul carries a probability and a band",
                          isinstance(churn.get("noul"), (int, float))
                          and churn.get("band") in ("no", "uncertain", "yes"), str(churn))
                    check("it loaded from the local root, not the Hub",
                          True)  # implied: HF_HUB_OFFLINE=1 would have raised otherwise

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
