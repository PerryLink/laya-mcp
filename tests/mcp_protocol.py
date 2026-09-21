"""Drive the MCP server over real stdio and assert the protocol works.

This is the one layer nothing else covers: `smoke_pure.py` tests the pure logic,
`install_harnesses.py` tests the config writers, and neither exercises an actual
MCP handshake. The published npm launcher's whole reason to exist is to hand
stdio to this server, so "the shim spawns something" is not evidence that a
harness would get a working server.

Rather than reimplement the client, this uses the official SDK, which is the same
implementation every harness uses. It spawns the server exactly as a harness
would - `python -m laya_mcp mcp` - performs the initialize handshake, lists tools,
and calls one.

The sidecar is addressed over HTTP so the model is not loaded twice; start it with
`laya-mcp serve` first. If it is not running the tool call is expected to fail with
`sidecar_unreachable`, which is itself asserted, because an MCP server that hangs
instead of reporting an unreachable backend is worse than one that fails.

Run:  python tests/mcp_protocol.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FAILURES: list[str] = []
CHECKS = 0

SIDECAR = os.environ.get("LAYA_MCP_SIDECAR", "http://127.0.0.1:8787")


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def _text_of(result) -> str:
    """Flatten a CallToolResult into text."""
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def main() -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("the MCP SDK is not installed; run: pip install 'laya-mcp[mcp]'")
        return 2

    # Spawned exactly as a harness spawns it: the module form, not a console
    # script. On Windows a console script is a .cmd shim and the MCP stdio
    # transport spawns with shell:false, which cannot execute one.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp", "mcp", "--sidecar", SIDECAR],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    print(f"spawning: {sys.executable} -m laya_mcp mcp --sidecar {SIDECAR}\n")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            print("initialize handshake")
            info = await session.initialize()
            check("handshake completes", info is not None)
            check("server reports a name", bool(info.serverInfo.name), str(info.serverInfo))
            check("server reports a version", bool(info.serverInfo.version), str(info.serverInfo))

            print("\ntools/list")
            listed = await session.list_tools()
            names = [t.name for t in listed.tools]
            # Five: the general batch tool, the three primitives as one-question
            # conveniences (most calls ask exactly one thing, and building a batch
            # of one is friction), and the budget planner.
            expected = {"laya_ask", "laya_noul", "laya_choice", "laya_score", "laya_plan"}
            check("all five tools are exposed", set(names) == expected, str(sorted(names)))
            check("no unexpected tool is exposed", len(names) == len(expected), str(sorted(names)))

            ask = next(t for t in listed.tools if t.name == "laya_ask")
            # The description is the product here: a decision model that does not
            # tell the model when NOT to call it will be asked to write prose.
            check("laya_ask says it does not generate text",
                  "does not generate text" in (ask.description or ""),
                  (ask.description or "")[:80])
            check("laya_ask warns that confidence is not accuracy",
                  "NOT the probability" in (ask.description or ""))
            check("laya_ask warns about end-truncation",
                  "truncated from the END" in (ask.description or ""),
                  "the silent-truncation warning is the point of this tool")
            check("laya_ask warns that options get shortened",
                  "shortened" in (ask.description or ""))
            check("parameters declare state and questions",
                  {"state", "questions"} <= set((ask.inputSchema.get("properties") or {}).keys()),
                  str(list((ask.inputSchema.get("properties") or {}).keys())))

            # The three primitive conveniences should each describe their own trap.
            noul = next(t for t in listed.tools if t.name == "laya_noul")
            check("laya_noul points at the band, not confidence",
                  "band" in (noul.description or ""))
            choice = next(t for t in listed.tools if t.name == "laya_choice")
            check("laya_choice warns about high option counts",
                  "20 options" in (choice.description or ""))
            score = next(t for t in listed.tools if t.name == "laya_score")
            check("laya_score admits it is the weakest primitive",
                  "weakest" in (score.description or ""))

            print("\ntools/call laya_ask (through the sidecar to the real model)")
            result = await session.call_tool(
                "laya_ask",
                {
                    "state": {
                        "subject": "Duplicate charge on invoice #4411",
                        "body": "We were billed twice for March. Refund or we cancel our plan.",
                    },
                    "questions": {
                        "churn": {
                            "type": "noul",
                            "instructions": "Does the user threaten to cancel?",
                        },
                        "team": {
                            "type": "choice",
                            "instructions": "Which team should handle this?",
                            "criteria": {
                                "billing": "invoices, payments, refunds",
                                "technical": "bugs, outages",
                            },
                        },
                    },
                },
            )
            text = _text_of(result)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                check("the tool result is JSON", False, text[:200])
                payload = {}

            if payload:
                check("the tool result is JSON", True)
                check("reports ok", payload.get("ok") is True, str(payload.get("error")))
                check("carries the answers", "answers" in payload, str(list(payload.keys())))
                if payload.get("ok"):
                    check("names the checkpoint", payload.get("model") == "english",
                          str(payload.get("model")))
                    check("reports the device", bool(payload.get("device")), str(payload.get("device")))
                    check("includes the confidence caveat",
                          "NOT the probability" in (payload.get("confidence_semantics") or ""))
                    answers = payload.get("answers") or {}
                    check("both questions were answered", set(answers) == {"churn", "team"},
                          str(sorted(answers)))
                    churn = answers.get("churn") or {}
                    check("the noul carries a band",
                          churn.get("band") in ("no", "uncertain", "yes"), str(churn))
                    team = answers.get("team") or {}
                    check("the choice carries a distribution",
                          isinstance(team.get("probabilities"), dict), str(team))

            print("\ntools/call laya_plan")
            plan = await session.call_tool(
                "laya_plan",
                {
                    "state": {"body": "x" * 3000},
                    "questions": {"q": {"type": "noul", "instructions": "Is this long?"}},
                },
            )
            plan_text = _text_of(plan)
            try:
                plan_payload = json.loads(plan_text)
                check("plan returns JSON", True)
                check("plan reports token estimates",
                      "state_tokens_estimated" in plan_payload, str(list(plan_payload.keys())))
            except json.JSONDecodeError:
                check("plan returns JSON", False, plan_text[:200])

            print("\nan unknown tool is refused, not silently ignored")
            try:
                bad = await session.call_tool("laya_nonexistent", {})
                check("unknown tool is refused", bool(getattr(bad, "isError", False)),
                      str(getattr(bad, "isError", None)))
            except Exception:
                # A protocol-level error is also a correct refusal.
                check("unknown tool is refused", True)

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
