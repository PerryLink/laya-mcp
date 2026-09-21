# laya-mcp

[Laya](https://github.com/NandhaKishorM/laya) is a fast, non-autoregressive
"System 1" decision model: it answers typed questions — `noul` (yes/no),
`choice`, `score` — over a piece of state and returns probabilities, in a single
forward pass. It is genuinely good, and it is a research artifact.

This is the part that makes it survive contact with a server.

```bash
pip install 'laya-mcp[mcp]'
laya-mcp serve            # loads the model once, keeps it warm on 127.0.0.1:8787
laya-mcp install          # registers it with whichever agent harness you have
```

> **Status: 0.1.0, work in progress.** The core is implemented and its pure logic
> is covered by 64 checks, but it has not yet been exercised end-to-end against a
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
| **Token-budget preflight** | `laya_plan` reports exactly what would be truncated, and how many tokens each option actually gets, **without running the model**. The option arithmetic reproduces `build_sequence` line for line. |
| **Structured errors** | Every Laya failure becomes a code, an HTTP status, the offending question id, and a hint. `KeyError('ranking')` becomes `invalid_question` naming the type. |
| **Honest confidence** | `confidence` is labelled for what it is, on every response. `noul` answers also carry a `no` / `uncertain` / `yes` band, because a calibrated probability is not a decision. |
| **Calibration store** | Fit a temperature per `(primitive, option bucket)` against your own labels, persist it, reload it. `laya-multilingual` ships **no** fitted temperatures at all, so its probabilities are raw until you do this. |
| **Device honesty** | Reports a silent CPU demotion, and `doctor` proves the GPU works by running a real op rather than trusting `torch.cuda.is_available()`. |
| **Serialised inference** | A lock, by default. Laya is not thread-safe: `system_one` reassigns `self.device` and calls `self.model.to(...)` on an OOM, so concurrent calls can race a device move against a forward pass. |
| **Restartable** | `DELETE /model` releases the model and empties the CUDA allocator cache, which `Router.unload` does not. A model server that leaks needs to be recyclable. |

---

## One installer, five harnesses

There is no portable way to register an MCP server. Measured against real
installed harnesses, they disagree on the file, the format, and the key:

| harness | config | format | key |
|---|---|---|---|
| Claude Code | `~/.claude.json` | JSON | `mcpServers` |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.<name>]` |
| opencode | `~/.config/opencode/opencode.json[c]` | JSON | `mcp` |
| OpenClaw | `~/.openclaw/openclaw.json` | JSON | `mcp.servers` |
| Hermes | `~/.hermes/config.yaml` | YAML | `mcp_servers` |

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
  be invented; this package will not pretend to.
* **Position bias is real.** One published fixture run answered "A" on 46 of 50
  multiple-choice items.
* **Accuracy falls off above ~20 options**, per the author.

Calibration makes a probability *honest*; it cannot make a model *right*. If the
accuracy is not there for your task, fit on your own domain or do not deploy it.

---

## Verify

```bash
python tests/smoke_pure.py         # 66 checks: validation, planning, calibration, errors
python tests/install_harnesses.py  # 35 checks: every harness dialect, in a temp dir
python tests/mcp_protocol.py       # 25 checks: a real MCP handshake and real tool calls
laya-mcp doctor                    # what is installed, and what the GPU can really do
```

126 checks, and each suite covers a layer the others cannot reach.

`smoke_pure.py` needs no torch, model, network or harness config. `install_harnesses.py`
redirects every harness into a temporary directory, because `~/.claude.json` is a
large shared file holding history and per-project state and a test that clobbered
it would be a worse bug than any it could catch.

`mcp_protocol.py` is the one that matters most and the one that was missing
longest. It spawns the server exactly as a harness does
(`python -m laya_mcp mcp`), performs the real `initialize` handshake with the
official SDK, lists tools, and calls them through the sidecar to the model. Two
real defects escaped the other suites and were caught only here: `FastMCP` in
mcp 1.30 takes no `version` argument, so the server failed to start at all; and
the token-budget warning was written into the DSH plugin's tool description but
never into this server's, so a client using MCP could not have known that an
oversized state is cut from the end.

The harness dialects were additionally verified by letting the harnesses parse
the files this tool writes: `codex mcp list --json` and `openclaw mcp list --json`
both report the registered server with the correct stdio transport. For the other
three the format is verified but harness acceptance is not, which is stated rather
than implied — `pi` has no MCP support at all, and this machine's Hermes runtime is
incomplete.

## Licence

Apache-2.0. Laya is Apache-2.0, by Convai Innovations. This is an independent
integration and is not affiliated with or endorsed by that project.
