# laya-mcp

[Laya](https://github.com/NandhaKishorM/laya) is a fast, non-autoregressive
"System 1" decision model: it answers typed questions — `noul` (yes/no),
`choice`, `score` — over a piece of state and returns probabilities, in a single
forward pass. It is genuinely good, and it is a research artifact.

This is the part that makes it survive contact with a server.

[English](README.md) · [简体中文](README-zh.md) · [Español](README-es.md) · [Português](README-pt.md) · [हिन्दी](README-hi.md)

```bash
pip install 'laya-mcp[mcp]'
laya-mcp serve            # loads the model once, keeps it warm on 127.0.0.1:8787
laya-mcp install          # registers it with whichever agent harness you have
```

> **Status: 0.2.2, work in progress.** The core is implemented and its pure logic
> is covered by 94 checks, but it has not yet been exercised end-to-end against a
> live harness in CI. Interfaces may move before 1.0.

---

## The problem this solves

Laya truncates things silently, and the omissions are the kind you only notice
after acting on a wrong answer.

**It cuts the state, from the end, and says nothing.** `build_sequence` gives the
state whatever room is left after the options and slices it `st[:room]`. A long
document loses its tail — which for a contract, a log or an email thread is often
where the answer was — and the model then answers about the surviving prefix at
full confidence. Nothing in the response marks it.

**It shortens options until labels are indistinguishable.** Options share a fixed
`head_max_len` budget (192 tokens on the English checkpoint, 256 on the others).
Past a point every label gets ~4 tokens. This is the documented cause of the
Banking77 collapse (0.425 against Jev's 0.870), and again, nothing reports it.

**Its validation is one check.** An unknown `type` is a bare `KeyError` from
`QTYPES[q["t"]]`; a missing `criteria` is a bare `KeyError` too. They are
indistinguishable from a bug in the model, and neither names the question at
fault.

**Its `confidence` is not accuracy.** It is `1 - H(p)/log(k)` for `choice` and
`score` and `max(p, 1-p)` for `noul`. A normalised entropy is *low* when
probability is spread out even when the top option is right, and *high* on a
confident wrong answer — the English checkpoint scores 0.000 accuracy on Khmer at
0.952 confidence. A threshold on it does not mean what it looks like.

**And it demotes itself to CPU in silence.** On a CUDA OOM it moves the model to
CPU in fp32, in place, permanently, printing to stdout. No flag is set anywhere.
A process that hits this once keeps answering, roughly 10–15x slower, and nothing
in the response admits it.

So this package adds what is missing: a **preflight** that says what would be
cut, **structured errors** naming the question, an **honest confidence contract**,
and a **health surface** that reports a demotion.

---

## What it does

| | |
|---|---|
| **Warm sidecar** | One `Router`, preloaded, for the life of the process. Laya's default (`max_loaded=1`) rebuilds a model on every language switch — measured upstream at a 7.4 s median reload on CPU, 10.3 s on a T4. |
| **Token-budget preflight** | `laya_plan` reports exactly what would be truncated, and how many tokens each option actually gets, **without running the model**. The option arithmetic reproduces `build_sequence` line for line, and the state figure is counted with **the checkpoint's own tokenizer** rather than estimated from characters. That last part is a fix, not a feature that always worked: `plan_questions` accepted a `tokenizer` argument from the start and **no caller ever passed one**, so every response reported `exact: false` and the character estimate alone decided `fits` — and that estimate is safe on prose but **under-reserves on exactly the inputs a judge is given**: JSON 1.45×, source code 1.08×, CJK 2.1×, CSV 2.15×. On those it could answer `fits: true` for a state the model then silently truncated, which is the failure this tool exists to prevent. The estimator's language guard was wrong in two further ways, both fixed: it averaged over the first 4,000 characters only, so a document that opens in English and continues in Chinese was measured with the Latin ratio (a wholly Chinese state of 239–830 characters was under-reserved by 2.5×), and `json.dumps`' default `ensure_ascii=True` turned CJK into `u`/`e`/`d`/`f` escapes that read as Latin (2.4× under-reserve). `tests/preflight_contract.py` now holds all of it. |
| **Structured errors** | Every Laya failure becomes a code, an HTTP status, the offending question id, and a hint. `KeyError('ranking')` becomes `invalid_question` naming the type. |
| **Honest confidence** | `confidence` is labelled for what it is, on every response. `noul` answers also carry a `no` / `uncertain` / `yes` band, because a calibrated probability is not a decision. |
| **Calibration store** | Fit a temperature per `(primitive, option bucket)` against your own labels, persist it, reload it. `laya-multilingual` ships **no** fitted temperatures at all, so its probabilities are raw until you do this. |
| **Device honesty** | Reports a silent CPU demotion, and `doctor` proves the GPU works by running a real op rather than trusting `torch.cuda.is_available()`. |
| **Serialised inference** | A lock, by default. Laya is not thread-safe: `system_one` reassigns `self.device` and calls `self.model.to(...)` on an OOM, so concurrent calls can race a device move against a forward pass. |
| **Restartable** | `DELETE /model` releases the model and empties the CUDA allocator cache, which `Router.unload` does not. A model server that leaks needs to be recyclable. |
| **A `noul` that is not a constant** | Laya renders every `noul` as `false: ...` / `true: ...` and then answers "false" to essentially all of them — 40 of 40 items, both languages, exactly chance. The label word is what breaks it, not the primitive, so a `noul` that carries a boundary is asked as a two-option choice under neutral labels and read back as `P(true)`: **0.500 → 1.000** (English) and **0.975** (multilingual) on the same forty items. A `noul` with no boundary is sent unchanged and the response says why. |

---

## One installer, six harnesses

There is no portable way to register an MCP server. Measured against real
installed harnesses, they disagree on the file, the format, and the key:

| harness | config | format | key |
|---|---|---|---|
| Claude Code | `~/.claude.json` | JSON | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | JSON | `mcpServers` |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.<name>]` |
| opencode | `~/.config/opencode/opencode.json[c]` | JSON | `mcp` |
| OpenClaw | `~/.openclaw/openclaw.json` | JSON | `mcp.servers` |
| Hermes | `HERMES_HOME`, else `%LOCALAPPDATA%\hermes` on Windows or `~/.hermes` | YAML | `mcp_servers` |

`laya-mcp install` detects which are present and writes the right shape to each.
Every writer merges rather than replaces, backs the file up first, and refuses to
touch a file it cannot parse — `~/.claude.json` is a large shared file holding
history and per-project state, and clobbering it to install a decision model
would be a catastrophic trade.

Two honest limitations:

* **opencode differs from everyone three more times** inside its own entry:
  `command` is a single array holding the executable *and* its arguments, the
  environment key is `environment`, not `env`, and the toggle is `enabled`.
  Setting `disabled: true` there is silently ignored.
* **`pi` is not supported.** Not an oversight: `pi` has no native MCP support.
  Its settings reference contains no MCP key, and its own upstream request for MCP
  is titled *"Add MCP extension example"* — in `pi`, MCP is an extension you
  build. There is no config file an installer can write. `install` detects it and
  says so.

`install` points the harness at `python -m laya_mcp mcp` rather than at the
`laya-mcp` console script, deliberately: on Windows a console script is a `.cmd`
shim and the MCP SDK spawns with `shell: false`, which cannot execute it.

---

## Tools

| tool | what it answers |
|---|---|
| `laya_ask` | A batch of typed questions over one state. The general one. |
| `laya_noul` | One yes/no question. Returns `P(true)` and a band. |
| `laya_choice` | One multiple-choice question. Returns the label and the distribution. |
| `laya_score` | One ordered-scale question. |
| `laya_plan` | "Will this fit, and what will be cut?" — no forward pass. |

Every description says what the tool is *not* for. A decision model asked to write
prose produces nothing useful, and an agent that does not know that will keep
trying.

---

## HTTP

```bash
laya-mcp serve --model english --port 8787
curl -s localhost:8787/health
curl -s localhost:8787/ask -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge", "body": "Billed twice. Refund or we cancel."},
  "questions": {
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
    "team":  {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages"}}
  }
}'
```

`GET /health`, `GET /capabilities`, `GET /version`, `POST /ask`, `POST /plan`,
`DELETE /model`. Loopback only by default; binding elsewhere warns loudly, because
there is no authentication.

`POST /plan` takes the same body as `/ask` and returns the same `budget` block
`/ask` reports, computed by the same `plan_questions` call — without a forward
pass. It is how a client can ask "will this be cut?" before paying for an answer.
On a cold host it pays a model *load*, which is not the same thing as an
inference.

---

## Configuration worth knowing

| flag | why |
|---|---|
| `--head-max-len` | Raised at startup, this is the fix for high-cardinality `choice`. Options share it, so more room per label is the only way to keep them distinguishable. Read fresh on every call, so setting it once is enough. |
| `--max-len` | The total budget. Raising it is the fix for a truncated state. |
| `--truncate-left` | Keep the **tail** of an oversized state instead of its head. Off by default because it changes which part of a long document the model reads — and it decides answers: one 16 958-character state with a decoy at the front and the correction at the back scored a `noul` **0.0706** with the front kept and **0.8341** with the tail kept. Use it when the end is where the answer is (a thread, a log, a contract's closing terms), and read `truncated.state.kept` to see which end survived. |
| `--concurrency` | Raise only if you know Laya is not sharing device state. The default of 1 is correctness, not caution. |
| `--sidecar` | Point `laya-mcp mcp` at a running `serve`. Strongly recommended: a harness spawns one stdio server per session, and hosting the model in each one pays the load cost per session. |

---

## What this does not fix

Upstream's own numbers are worth repeating, because an integration layer that
implies otherwise is lying to you.

* **The base checkpoints are near chance zero-shot on typed decisions** — 0.362
  for English against a **0.461 majority-class baseline**. Guessing the most
  common answer beats the model.
* **`score` is the weakest primitive.** Independently measured at 35% against
  Jev's 70% on a five-level ordinal task.
* **Calibration needs labelled data.** Raw ECE is 0.466 for English and 0.314 for
  multilingual, improving to 0.081 and 0.106 after fitting. A temperature cannot
  be invented; this package will not pretend to. A fit that runs into the edge of
  its search grid is recorded in `saturated_buckets` as a bound rather than
  returned as a temperature, because a bound describes the sample, not the model.
* **Position bias is real.** One published fixture run answered "A" on 46 of 50
  multiple-choice items.
* **Accuracy falls off above ~20 options**, per the author.

Calibration makes a probability *honest*; it cannot make a model *right*. If the
accuracy is not there for your task, fit on your own domain or do not deploy it.

---

## Verify

```bash
python tests/smoke_pure.py         # 94 checks: validation, planning, calibration, errors
python tests/install_harnesses.py  # 38 checks: every harness dialect, in a temp dir
python tests/mcp_protocol.py       # 37 checks live (32 offline): a real MCP handshake and real tool calls
python tests/stdio_latency.py      # handshake <5 s, tools/list instant, tools/call returns
python tests/language_probe.py     # what each checkpoint can actually do, per language
laya-mcp doctor                    # what is installed, and what the GPU can really do
```

169 checks in the three suites, and each covers a layer the others cannot reach.
`stdio_latency.py` and `language_probe.py` need a model and are measurements
rather than assertions, so they are run by hand and their numbers are quoted
above.

`smoke_pure.py` needs no torch, model, network or harness config. `install_harnesses.py`
redirects every harness into a temporary directory, because `~/.claude.json` is a
large shared file holding history and per-project state and a test that clobbered
it would be a worse bug than any it could catch.

`mcp_protocol.py` is the one that matters most and the one that was missing
longest. It spawns the server exactly as a harness does
(`python -m laya_mcp mcp`), performs the real `initialize` handshake with the
official SDK, lists tools, and calls them. Two real defects escaped the other
suites and were caught only here: `FastMCP` in mcp 1.30 takes no `version`
argument, so the server failed to start at all; and the token-budget warning was
written into the DSH plugin's tool description but never into this server's, so a
client using MCP could not have known that an oversized state is cut from the end.

Acceptance was then verified by letting the harnesses parse and *connect to* the
files this tool writes, which is the only test that distinguishes a written file
from an accepted one:

| harness | how | result |
|---|---|---|
| opencode | `opencode mcp list` | ✓ connected |
| claude | `claude mcp list` | √ Connected |
| cursor | Settings > Tools & MCP (full quit-and-reopen required) | entry under `mcpServers` in `~/.cursor/mcp.json`; project `.cursor/mcp.json` wins with no merge |
| codex | `codex mcp list --json` | `enabled`, `"type": "stdio"`, correct argv — `auth_status: unsupported` is not a fault, a local stdio server needs none |
| OpenClaw | `openclaw mcp list --json` | reports the server, stdio transport |
| Hermes | `hermes mcp list` | ✓ enabled, and `hermes mcp test laya` connects and finds all 5 tools |
| `pi` | — | no native MCP support; `install` detects it and says so |

Every harness this installer supports now confirms through its own tooling. That
is the only check that distinguishes a written file from an accepted one, and it
earns its keep: Hermes reads its config from `%LOCALAPPDATA%\hermes` on Windows,
not `~/.hermes`, so the installer had been reporting success while writing a file
nothing read. Two faults hid it — Hermes was marked unverifiable, and on Windows
none of these listers could even be launched, because npm ships each of them as a
`.cmd` shim that `CreateProcess` refuses to execute.

That table is the reason `stdio_latency.py` exists. Every harness above gives an
MCP server 30 seconds to finish `initialize`, and answering the handshake only
after loading a checkpoint took 19 s uncontended and 275 s while another model
held the GPU — so all of them reported "Failed to connect" on a config they had
parsed perfectly. The load now runs behind the handshake.

## Licence

Apache-2.0. Laya is Apache-2.0, by Convai Innovations. This is an independent
integration and is not affiliated with or endorsed by that project.
