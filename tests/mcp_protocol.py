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

So this runs in two modes and asserts whichever contract applies:

* **sidecar up** - the handshake, the tool list, and real answers carried back
  through the protocol, with the codes and caveats intact.
* **sidecar down** - the same handshake and tool list (neither needs the model),
  plus the structured refusal: a code naming the cause and a hint naming the fix.

The second mode is what makes this suite runnable in CI. Nothing else covers the
protocol layer, and a layer covered only on a machine that happens to have a GPU
and a warm checkpoint is a layer that is not covered.

Run:  python tests/mcp_protocol.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FAILURES: list[str] = []
CHECKS = 0

SIDECAR = os.environ.get("LAYA_MCP_SIDECAR", "http://127.0.0.1:8787")


def sidecar_reachable(url: str) -> bool:
    """Whether a warm sidecar is answering. Decides which half of this suite runs."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any failure at all means "not reachable"
        return False


LIVE = sidecar_reachable(SIDECAR)


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

    # The direction is a server setting now, so the budget sentence is built rather
    # than fixed. All three variants are checked directly, because this suite only
    # ever spawns the `--sidecar` one and the other two would go unexercised.
    from laya_mcp.cli import _parse_filter
    from laya_mcp.mcp_server import TOOL_NAMES, _budget_note, normalise_tool_filter

    note_front = _budget_note(False)
    note_tail = _budget_note(True)
    note_sidecar = _budget_note(None)
    check("a front-keeping server says it discards the tail",
          "discards the tail" in note_front and "FRONT" in note_front, note_front[:130])
    check("a tail-keeping server says it discards the front",
          "discards the front" in note_tail and "TAIL" in note_tail, note_tail[:130])
    check("a sidecar-backed server points at the field instead of guessing",
          "truncated.state.kept" in note_sidecar and "discards" not in note_sidecar,
          note_sidecar[:170])
    check("every variant still carries the budget warning",
          all("token budget" in n and "shortened" in n
              for n in (note_front, note_tail, note_sidecar)))

    # `--filter` was documented with short names and validated against registered
    # ones, so the documented spelling was rejected outright - and `build_server`,
    # which registers by matching names, silently produced a server with no tools.
    # Untested until the DSH bundle was pointed at it, which is why these exist.
    expected_tools = {"laya_noul", "laya_choice", "laya_score"}
    for spelling in ("noul,choice,score", "laya_noul,laya_choice,laya_score",
                     " noul , choice , score "):
        resolved = normalise_tool_filter(_parse_filter(spelling))
        check(f"--filter {spelling.strip()!r} resolves to the registered names",
              set(resolved or ()) == expected_tools, str(sorted(resolved or ())))
    check("no --filter means every tool", normalise_tool_filter(None) is None)
    check("the filter names are all real tools",
          expected_tools <= set(TOOL_NAMES), str(TOOL_NAMES))

    from laya_mcp.errors import LayaMcpError
    from laya_mcp.mcp_server import Backend, build_server
    from laya_mcp.worker import WorkerConfig

    for bad, label in ((("laya_typo",), "an unknown tool name"), ((), "an empty filter")):
        try:
            build_server(Backend(sidecar="http://127.0.0.1:9", config=WorkerConfig()), bad)
            check(f"--filter with {label} is refused rather than exposing nothing", False)
        except LayaMcpError:
            check(f"--filter with {label} is refused rather than exposing nothing", True)

    # Spawned exactly as a harness spawns it: the module form, not a console
    # script. On Windows a console script is a .cmd shim and the MCP stdio
    # transport spawns with shell:false, which cannot execute one.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp", "mcp", "--sidecar", SIDECAR],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    print(f"mode: {'LIVE' if LIVE else 'OFFLINE'} - sidecar "
          f"{'reachable' if LIVE else 'NOT reachable'} at {SIDECAR}")
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
            description = ask.description or ""
            check("laya_ask warns that an oversized state is truncated",
                  "truncated" in description and "token budget" in description,
                  description[:120])
            # Which end is at risk depends on how the server was started, so the
            # description either names it or points at the field that does. Asserting
            # the substance rather than one wording: the literal "truncated from the
            # END" stopped being true the moment `--truncate-left` existed, and a
            # description that kept saying it would be worse than saying nothing.
            check("laya_ask names the end that survives, or where to read it",
                  "discards the tail" in description or "discards the front" in description
                  or "truncated.state.kept" in description,
                  description[:220])
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
                if not LIVE:
                    # No sidecar. Nothing about the model can be asserted, but the
                    # refusal can be: a code naming the cause, and a hint naming
                    # the command that fixes it.
                    check("refuses instead of hanging", payload.get("ok") is False,
                          str(sorted(payload)))
                    check("names the unreachable sidecar",
                          payload.get("error") == "sidecar_unreachable",
                          str(payload.get("error")))
                    check("says what to run",
                          "laya-mcp serve" in (payload.get("hint") or ""),
                          str(payload.get("hint")))
                else:
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
                if LIVE:
                    check("plan reports token estimates",
                          "state_tokens_estimated" in plan_payload, str(list(plan_payload.keys())))
                else:
                    check("plan refuses with the sidecar down",
                          plan_payload.get("error") == "sidecar_unreachable",
                          str(plan_payload.get("error")))
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
