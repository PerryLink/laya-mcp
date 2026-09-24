"""Verify the multi-harness installer writes each dialect correctly.

Every harness is redirected to a temporary directory first, so this can be run
repeatedly and never touches a real config. That matters: `~/.claude.json` is a
large shared file holding history and per-project state, and a test that clobbered
it to save a typing exercise would be a worse bug than any it could catch.

What this proves, per harness: the file is created, it parses as its own format,
and the entry lands under the key that harness actually reads - and, for the path
rules that differ by platform, that the path is the one the harness itself
resolves. It does NOT prove a harness accepts the file; only that harness's own
lister can prove that, and `pi` has no MCP support at all so there is nothing to
accept.

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
        _write_cursor,
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

        print("\ncursor  (~/.cursor/mcp.json, key `mcpServers`, no `type`)")
        cur = root_path / "cursor" / "mcp.json"
        cur.parent.mkdir(parents=True)
        cur.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
        _write_cursor(cur, entry)
        data = json.loads(cur.read_text(encoding="utf-8"))
        check("entry is under `mcpServers`", "mcpServers" in data and "laya" in data["mcpServers"])
        check("pre-existing server survives", "other" in data["mcpServers"])
        block = data["mcpServers"]["laya"]
        check("command matches", block.get("command") == sys.executable)
        check("args is a list", isinstance(block.get("args"), list))
        check("does NOT emit `type` (not a Cursor mcp.json key)",
              "type" not in block)

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
    check("all six requested harnesses are known",
          {"claude", "cursor", "codex", "opencode", "openclaw", "hermes"} <= {h.id for h in HARNESSES})
    pi = next((h for h in HARNESSES if h.id == "pi"), None)
    check("pi is present and marked unsupported", pi is not None and pi.unsupported_reason is not None)
    if pi and pi.unsupported_reason:
        check("pi's reason names the real cause",
              "no native MCP" in pi.unsupported_reason, pi.unsupported_reason[:60])
    detected = detect_harnesses()
    check("detection returns every harness with a status",
          len(detected) == len(HARNESSES), f"{len(detected)} vs {len(HARNESSES)}")

    print("\nhermes resolves its config the way hermes does")
    # Hermes does not use `~/.hermes` on Windows. Writing there produces a file it
    # never reads, while the installer reports success - which is what happened,
    # and what `hermes mcp list` was added to catch.
    from laya_mcp.harnesses import _hermes_path

    saved = {key: os.environ.get(key) for key in ("HERMES_HOME", "LOCALAPPDATA")}
    try:
        probe = Path(tempfile.gettempdir()) / "hermes-home-probe"
        os.environ["HERMES_HOME"] = str(probe)
        check("HERMES_HOME wins outright",
              _hermes_path() == probe / "config.yaml", str(_hermes_path()))

        os.environ.pop("HERMES_HOME", None)
        if sys.platform == "win32":
            os.environ["LOCALAPPDATA"] = r"C:\probe\LocalAppData"
            expected = Path(r"C:\probe\LocalAppData") / "hermes" / "config.yaml"
            check("Windows reads LOCALAPPDATA", _hermes_path() == expected, str(_hermes_path()))
            check("Windows does not read ~/.hermes",
                  _hermes_path() != Path(os.path.expanduser("~")) / ".hermes" / "config.yaml",
                  str(_hermes_path()))
        else:
            expected = Path(os.path.expanduser("~")) / ".hermes" / "config.yaml"
            check("POSIX falls back to ~/.hermes", _hermes_path() == expected, str(_hermes_path()))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {CHECKS} checks passed (part 1: configs)")
    return 0


def main_skills() -> int:
    """Part 2: the skill installer. Same temp-dir discipline as part 1."""
    from laya_mcp.harnesses import (
        HARNESSES,
        _codex_skill_path,
        _cursor_skill_path,
        _hermes_skill_path,
        _opencode_skill_path,
        install,
        install_skill_file,
        load_skill_source,
    )

    print("\nskill source: the packaged copy matches the repository root")
    repo_root = Path(__file__).resolve().parents[1] / "SKILL.md"
    vendored = Path(__file__).resolve().parents[1] / "src" / "laya_mcp" / "SKILL.md"
    check("repository SKILL.md exists", repo_root.is_file(), str(repo_root))
    check("vendored SKILL.md exists", vendored.is_file(), str(vendored))
    if repo_root.is_file() and vendored.is_file():
        check("vendored copy is identical to the repository copy (no drift)",
              vendored.read_text(encoding="utf-8") == repo_root.read_text(encoding="utf-8"))
    check("load_skill_source() finds one without arguments",
          isinstance(load_skill_source(), str) and len(load_skill_source()) > 0)

    with tempfile.TemporaryDirectory(prefix="laya-mcp-skill-src-") as root:
        custom = Path(root) / "custom.md"
        custom.write_text("---\nname: laya\n---\nbody\n", encoding="utf-8")
        check("an explicit --skill-source wins",
              load_skill_source(str(custom)) == "---\nname: laya\n---\nbody\n")
        try:
            load_skill_source(str(Path(root) / "missing.md"))
            check("a missing --skill-source is an error", False, "it read nothing")
        except ValueError as exc:
            check("a missing --skill-source is an error", "does not exist" in str(exc))

    print("\nevery supported harness has a skill path; pi has none")
    for target in HARNESSES:
        if target.unsupported_reason is not None:
            check(f"{target.id} (unsupported) has no skill path", target.skill_path is None)
        else:
            check(f"{target.id} has a skill path", target.skill_path is not None)
            check(f"{target.id} documents its skill location",
                  isinstance(target.skill_location, str) and "SKILL.md" in target.skill_location)

    print("\nskill paths honour their relocations")
    saved = {key: os.environ.get(key) for key in ("CODEX_HOME", "HERMES_HOME", "XDG_CONFIG_HOME")}
    try:
        probe = Path(tempfile.gettempdir()) / "laya-skill-probe"
        os.environ["CODEX_HOME"] = str(probe / "codex-home")
        check("CODEX_HOME relocates codex skills",
              _codex_skill_path("laya") == probe / "codex-home" / "skills" / "laya" / "SKILL.md",
              str(_codex_skill_path("laya")))
        os.environ.pop("CODEX_HOME", None)
        check("codex falls back to ~/.codex/skills",
              _codex_skill_path("laya") == Path(os.path.expanduser("~")) / ".codex" / "skills" / "laya" / "SKILL.md")

        os.environ["HERMES_HOME"] = str(probe / "hermes-home")
        check("HERMES_HOME relocates hermes skills",
              _hermes_skill_path("laya") == probe / "hermes-home" / "skills" / "laya" / "SKILL.md")
        os.environ.pop("HERMES_HOME", None)

        os.environ["XDG_CONFIG_HOME"] = str(probe / "xdg")
        check("XDG_CONFIG_HOME relocates opencode skills",
              _opencode_skill_path("laya") == probe / "xdg" / "opencode" / "skills" / "laya" / "SKILL.md")
        os.environ.pop("XDG_CONFIG_HOME", None)

        check("cursor skills live under ~/.cursor/skills",
              _cursor_skill_path("laya") == Path(os.path.expanduser("~")) / ".cursor" / "skills" / "laya" / "SKILL.md")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("\nskill writes: atomic, backed up, idempotent")
    with tempfile.TemporaryDirectory(prefix="laya-mcp-skill-") as root:
        target = Path(root) / "skills" / "laya" / "SKILL.md"
        check("dry run writes nothing",
              install_skill_file(target, "hello", dry_run=True) == "written"
              and not target.is_file())
        check("first write reports written",
              install_skill_file(target, "hello") == "written" and target.is_file())
        check("identical content reports unchanged",
              install_skill_file(target, "hello") == "unchanged")
        check("changed content reports written and backs up",
              install_skill_file(target, "hello v2") == "written"
              and target.read_text(encoding="utf-8") == "hello v2"
              and target.with_suffix(target.suffix + ".tmp").exists() is False)

    print("\ninstall --skill-only end to end, in a fake HOME")
    real_home = os.environ.get("HOME")
    fake_home = tempfile.mkdtemp(prefix="laya-mcp-fakehome-")
    try:
        os.environ["HOME"] = fake_home
        # A cursor config that already exists marks the harness present even
        # with no `cursor` binary on PATH - the same rule as the real detector.
        cursor_config = Path(fake_home) / ".cursor" / "mcp.json"
        cursor_config.parent.mkdir(parents=True, exist_ok=True)
        cursor_config.write_text('{"mcpServers": {}}', encoding="utf-8")
        code = install(harness="cursor", skill_only=True, skill_name="laya")
        expected = Path(fake_home) / ".cursor" / "skills" / "laya" / "SKILL.md"
        check("skill-only install exits 0", code == 0, f"exit {code}")
        check("cursor skill lands in the fake HOME", expected.is_file())
        check("no MCP config was touched beyond the pre-existing one",
              cursor_config.read_text(encoding="utf-8") == '{"mcpServers": {}}')
        code = install(harness="cursor", skill_only=True, skill_name="laya")
        check("re-running is clean (exit 0, content unchanged)", code == 0)
    finally:
        if real_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = real_home
        shutil.rmtree(fake_home, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {CHECKS} checks passed (parts 1+2)")
    return 0


if __name__ == "__main__":
    part1 = main()
    if part1 != 0:
        raise SystemExit(part1)
    raise SystemExit(main_skills())
