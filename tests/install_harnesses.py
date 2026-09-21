"""Verify the multi-harness installer writes each dialect correctly.

Every harness is redirected to a temporary directory first, so this can be run
repeatedly and never touches a real config. That matters: `~/.claude.json` is a
large shared file holding history and per-project state, and a test that clobbered
it to save a typing exercise would be a worse bug than any it could catch.

What this proves, per harness: the file is created, it parses as its own format,
and the entry lands under the key that harness actually reads. It does NOT prove a
harness accepts the file - only that harness's own lister can prove that, and two
of the five are not installable here (pi has no MCP support at all, and this
machine's hermes runtime is incomplete).

Run:  python tests/install_harnesses.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

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


def main() -> int:
    from laya_mcp.harnesses import (
        HARNESSES,
        _write_claude,
        _write_codex,
        _write_hermes,
        _write_openclaw,
        _write_opencode,
        build_entry,
        detect_harnesses,
    )

    print("build_entry")
    # A real interpreter path, never a shim: python.exe spawns fine with
    # shell:false, and wrapping it in `cmd /c` would mangle a path with spaces.
    entry = build_entry("laya", python=sys.executable)
    check("command is the interpreter", entry.command == sys.executable, entry.command)
    check("invokes the module, not a console script",
          entry.args[:2] == ["-m", "laya_mcp"], str(entry.args))
    check("not wrapped in cmd on any platform",
          entry.command != "cmd", entry.command)

    # A .cmd shim DOES need wrapping on Windows, because the MCP SDK spawns with
    # shell:false and cannot execute a batch file. Measured: spawn("npx") is ENOENT
    # and spawn("npx.cmd") is EINVAL on Node 22.
    from laya_mcp.harnesses import Entry

    wrapped = Entry(name="x", command="npx", args=["-y", "pkg"]).windows_safe()
    if os.name == "nt":
        check("a .cmd shim is wrapped in cmd /c on Windows",
              wrapped.command == "cmd" and wrapped.args[:2] == ["/c", "npx"], str(wrapped.args))
    else:
        check("no wrapping off Windows", wrapped.command == "npx", wrapped.command)

    with tempfile.TemporaryDirectory(prefix="laya-mcp-harness-") as root:
        root_path = Path(root)

        print("\nclaude  (~/.claude.json, key `mcpServers`)")
        claude = root_path / "claude" / ".claude.json"
        claude.parent.mkdir(parents=True)
        # Pre-existing unrelated state, to prove the merge preserves it.
        claude.write_text(
            json.dumps({"numStartups": 7, "projects": {"/x": {"mcpServers": {}}}}, indent=2),
            encoding="utf-8",
        )
        _write_claude(claude, entry)
        data = json.loads(claude.read_text(encoding="utf-8"))
        check("entry is under `mcpServers`", "mcpServers" in data and "laya" in data["mcpServers"])
        check("unrelated keys survive the merge", data.get("numStartups") == 7)
        check("project state survives", "/x" in data.get("projects", {}))
        block = data["mcpServers"]["laya"]
        check("declares type: stdio", block.get("type") == "stdio")
        check("does NOT emit alwaysAllow (not a real Claude MCP key)",
              "alwaysAllow" not in block)

        print("\ncodex  (~/.codex/config.toml, table `[mcp_servers.<name>]`)")
        codex = root_path / "codex" / "config.toml"
        codex.parent.mkdir(parents=True)
        codex.write_text('model = "gpt-5"\n\n[history]\npersistence = "save-all"\n', encoding="utf-8")
        _write_codex(codex, entry)
        text = codex.read_text(encoding="utf-8")
        check("table header is present", "[mcp_servers.laya]" in text)
        check("pre-existing config survives", 'model = "gpt-5"' in text and "[history]" in text)
        check("command is quoted TOML", f'command = "{sys.executable.replace(chr(92), chr(92) * 2)}"' in text
              or "command = " in text)
        # A Windows path must be escaped or TOML rejects the file outright.
        try:
            import tomllib

            parsed = tomllib.loads(text)
            check("the whole file still parses as TOML", True)
            check("parsed command matches", parsed["mcp_servers"]["laya"]["command"] == sys.executable)
            check("parsed unrelated model survives", parsed.get("model") == "gpt-5")
        except ImportError:
            print("  SKIP  tomllib unavailable")
        except Exception as exc:  # noqa: BLE001
            check("the whole file still parses as TOML", False, str(exc))

        # Idempotence: writing twice must replace, not append a duplicate table,
        # which is a TOML error and would break the user's config.
        _write_codex(codex, entry)
        check("re-writing does not duplicate the table",
              codex.read_text(encoding="utf-8").count("[mcp_servers.laya]") == 1)

        print("\nopencode  (key `mcp`, array command, `environment`, `enabled`)")
        oc = root_path / "opencode" / "opencode.jsonc"
        oc.parent.mkdir(parents=True)
        oc.write_text('{\n  // a comment, which plain JSON would reject\n  "$schema": "x",\n}\n', encoding="utf-8")
        _write_opencode(oc, entry)
        data = json.loads(oc.read_text(encoding="utf-8"))
        check("entry is under `mcp`", "mcp" in data and "laya" in data["mcp"])
        block = data["mcp"]["laya"]
        check("command is an ARRAY (opencode-only)", isinstance(block["command"], list))
        check("array holds executable then args",
              block["command"][0] == sys.executable and block["command"][1:3] == ["-m", "laya_mcp"])
        check("uses `enabled`, not `disabled`", block.get("enabled") is True and "disabled" not in block)
        check("pre-existing $schema survives", data.get("$schema") == "x")
        check("a --jsonc comment did not break parsing", True)

        print("\nopenclaw  (key `mcp.servers`)")
        ocl = root_path / "openclaw" / "openclaw.json"
        ocl.parent.mkdir(parents=True)
        ocl.write_text(json.dumps({"commands": {"x": 1}}), encoding="utf-8")
        _write_openclaw(ocl, entry)
        data = json.loads(ocl.read_text(encoding="utf-8"))
        check("entry is under `mcp.servers`", data["mcp"]["servers"]["laya"]["command"] == sys.executable)
        check("unrelated top-level state survives", data.get("commands") == {"x": 1})

        print("\nhermes  (YAML, key `mcp_servers`)")
        try:
            import yaml  # noqa: F401
        except ImportError:
            print("  SKIP  PyYAML not installed; hermes writing needs it")
        else:
            hm = root_path / "hermes" / "config.yaml"
            hm.parent.mkdir(parents=True)
            hm.write_text("model: claude\ntools:\n  - shell\n", encoding="utf-8")
            _write_hermes(hm, entry)
            data = yaml.safe_load(hm.read_text(encoding="utf-8"))
            check("entry is under `mcp_servers` (snake_case)", "mcp_servers" in data)
            block = data["mcp_servers"]["laya"]
            check("command matches", block["command"] == sys.executable)
            check("args is a list", isinstance(block["args"], list))
            check("enabled defaults true", block.get("enabled") is True)
            check("unrelated YAML survives", data.get("model") == "claude")

        print("\nrefusal: an unparseable config is left alone")
        broken = root_path / "broken" / "config.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{ this is not json", encoding="utf-8")
        try:
            _write_claude(broken, entry)
            check("refuses to rewrite a file it cannot parse", False, "it overwrote it")
        except ValueError as exc:
            check("refuses to rewrite a file it cannot parse", "not valid JSON" in str(exc))
            check("the broken file is untouched",
                  broken.read_text(encoding="utf-8") == "{ this is not json")

    print("\nharness registry")
    check("all five requested harnesses are known",
          {"claude", "codex", "opencode", "openclaw", "hermes"} <= {h.id for h in HARNESSES})
    pi = next((h for h in HARNESSES if h.id == "pi"), None)
    check("pi is present and marked unsupported", pi is not None and pi.unsupported_reason is not None)
    if pi and pi.unsupported_reason:
        check("pi's reason names the real cause",
              "no native MCP" in pi.unsupported_reason, pi.unsupported_reason[:60])
    detected = detect_harnesses()
    check("detection returns every harness with a status",
          len(detected) == len(HARNESSES), f"{len(detected)} vs {len(HARNESSES)}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
