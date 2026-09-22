# laya-mcp

[Laya](https://github.com/NandhaKishorM/laya) 是一个快速的非自回归「System 1」决策模型：对一段 state 提出有类型的问题——`noul`（是/否）、`choice`、`score`——在一次前向传播中返回概率。它是真正好用的，同时也是一个研究产物。

本包是让它经得起服务器环境的那一层。

[English](README.md) · [简体中文](README-zh.md) · [Español](README-es.md) · [Português](README-pt.md) · [हिन्दी](README-hi.md)

```bash
pip install 'laya-mcp[mcp]'
laya-mcp serve            # 加载一次模型，常驻在 127.0.0.1:8787
laya-mcp install          # 注册到本机已有的任意 agent harness
```

> **状态：0.2.1，开发中。** 核心已实现，纯逻辑部分由 85 项检查覆盖；但尚未在 CI 中经过真实 harness 的端到端验证。1.0 之前接口可能变动。

---

## 它解决的问题

Laya 会静默地截断东西，而这些省略恰恰是你在依据错误答案行动之后才会发现的。

**它会从末尾截断 state，而且不作声。** `build_sequence` 把选项之外的剩余空间给 state，然后切成 `st[:room]`。一份长文档因此丢掉尾部——而对合同、日志或邮件线程来说，答案往往就在那里——模型随后就着残存的前缀以满格置信度作答。响应里没有任何标记。

**它会把选项压缩到标签无法区分。** 选项共享固定的 `head_max_len` 预算（英文 checkpoint 192 token，其余 256）。超过某个点之后每个标签只剩约 4 个 token。这是有据可查的 Banking77 崩塌原因（0.425，对比 Jev 的 0.870），同样没有任何报告。

**它的校验只有一次检查。** 未知的 `type` 是来自 `QTYPES[q["t"]]` 的裸 `KeyError`；缺失的 `criteria` 也是裸 `KeyError`。两者与模型自身的 bug 无法区分，且都不指出是哪个问题出的错。

**它的 `confidence` 不是准确率。** 对 `choice` 和 `score` 它是 `1 - H(p)/log(k)`，对 `noul` 是 `max(p, 1-p)`。归一化熵在概率铺开时是*低*的——即使首选选项是对的；在自信地答错时是*高*的——英文 checkpoint 在高棉语上以 0.952 的置信度拿到 0.000 的准确率。对它设阈值，含义和看起来的不一样。

**而且它会静默降级到 CPU。** 遇到 CUDA OOM 时，它把模型就地在 fp32 下移到 CPU，永久性地，只往 stdout 打印一行。任何地方都没有标志位。命中一次的进程会继续作答，慢大约 10–15 倍，而响应里没有任何东西承认这件事。

所以本包补上缺失的部分：**预检**告诉你什么会被截断，**结构化错误**指出是哪个问题，**诚实的置信度契约**，以及报告降级的**健康面**。

---

## 它做什么

| | |
|---|---|
| **常驻 sidecar** | 一个 `Router`，预加载，存活于整个进程生命周期。Laya 的默认行为（`max_loaded=1`）会在每次语言切换时重建模型——上游实测 CPU 上中位重载 7.4 秒，T4 上 10.3 秒。 |
| **Token 预算预检** | `laya_plan` **不运行模型**就报告究竟什么会被截断、每个选项实际分到多少 token。选项部分的算术逐行复现 `build_sequence`。 |
| **结构化错误** | 每个 Laya 失败都变成错误码、HTTP 状态、出问题的问题 id，以及一条 hint。`KeyError('ranking')` 变成指出类型的 `invalid_question`。 |
| **诚实的置信度** | 每个响应都标注 `confidence` 究竟是什么。`noul` 答案还带 `no` / `uncertain` / `yes` 分带，因为标定过的概率不是决策。 |
| **标定存储** | 针对你自己的标注数据，按 `(primitive, 选项桶)` 拟合温度、持久化、重新加载。`laya-multilingual` **完全没有**出厂拟合温度，所以在做这件事之前它的概率是生的。 |
| **设备诚实** | 报告静默的 CPU 降级；`doctor` 通过运行一次真实算子来证明 GPU 可用，而不是相信 `torch.cuda.is_available()`。 |
| **串行化推理** | 默认加锁。Laya 不是线程安全的：`system_one` 在 OOM 时会重新赋值 `self.device` 并调用 `self.model.to(...)`，所以并发调用可能与设备迁移竞争。 |
| **可回收** | `DELETE /model` 释放模型并清空 CUDA 分配器缓存，这是 `Router.unload` 不做的。会泄漏的模型服务器需要能被回收。 |
| **一个不是常数的 `noul`** | Laya 把每个 `noul` 渲染为 `false: ...` / `true: ...`，然后几乎对所有输入都答「false」——两种语言各 40/40，正好是随机水平。坏掉的是标签词而不是 primitive，所以带 boundary 的 `noul` 会以中性标签当作两选项 choice 提问，再读回 `P(true)`：同样这 40 项上 **0.500 → 1.000**（英文）与 **0.975**（多语言）。不带 boundary 的 `noul` 原样发出，响应会说明原因。 |

---

## 一个安装器，五个 harness

注册 MCP server 没有可移植的做法。对着真实安装的 harness 实测，它们在文件、格式和键名上都不一样：

| harness | 配置 | 格式 | 键 |
|---|---|---|---|
| Claude Code | `~/.claude.json` | JSON | `mcpServers` |
| Codex | `~/.codex/config.toml` | TOML | `[mcp_servers.<name>]` |
| opencode | `~/.config/opencode/opencode.json[c]` | JSON | `mcp` |
| OpenClaw | `~/.openclaw/openclaw.json` | JSON | `mcp.servers` |
| Hermes | `HERMES_HOME`，否则 Windows 上是 `%LOCALAPPDATA%\hermes`、其余是 `~/.hermes` | YAML | `mcp_servers` |

`laya-mcp install` 检测哪些存在，并给每个写入正确的形状。所有写入器都是合并而非替换，先备份，并且拒绝对无法解析的文件动手——`~/.claude.json` 是一个装着历史与逐项目状态的大共享文件，为了装一个决策模型而覆盖它是灾难性的交易。

两个诚实的限制：

* **opencode 在自己的条目里还有三处不同**：`command` 是同时装着可执行文件和参数的单个数组；环境变量的键是 `environment` 而不是 `env`；开关是 `enabled`。在那里设 `disabled: true` 会被静默忽略。
* **不支持 `pi`。** 这不是疏忽：`pi` 没有原生 MCP 支持。它的设置文档里没有任何 MCP 键，它上游关于 MCP 的请求标题是*「Add MCP extension example」*——在 `pi` 里，MCP 是要你自己构建的扩展。没有配置文件可供安装器写入。`install` 会检测到并说明。

`install` 让 harness 指向 `python -m laya_mcp mcp` 而不是 `laya-mcp` 控制台脚本，这是有意的：在 Windows 上控制台脚本是一个 `.cmd` 垫片，而 MCP SDK 以 `shell: false` 生成进程，无法执行它。

---

## 工具

| 工具 | 它回答什么 |
|---|---|
| `laya_ask` | 对一段 state 的一批有类型问题。通用的那个。 |
| `laya_noul` | 一个是/否问题。返回 `P(true)` 与分带。 |
| `laya_choice` | 一个多选题。返回标签与分布。 |
| `laya_score` | 一个有序量表问题。 |
| `laya_plan` | 「放得下吗，什么会被截断？」——不跑前向传播。 |

每个描述都写明该工具*不*适用于什么。被要求写散文的决策模型产不出有用的东西，而不知道这一点的 agent 会一直尝试。

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

`GET /health`、`GET /capabilities`、`GET /version`、`POST /ask`、`POST /plan`、`DELETE /model`。默认只绑回环；绑到别处会大声警告，因为它没有认证。

`POST /plan` 接收与 `/ask` 相同的请求体，返回 `/ask` 报告的同一个 `budget` 块，由同一次 `plan_questions` 调用算出——**不跑前向传播**。客户端借此在为一个答案付费之前问「会截断吗」。在冷启动的主机上它需要付出一次模型*加载*，这与一次推理不是一回事。

---

## 值得知道的配置

| 参数 | 为什么 |
|---|---|
| `--head-max-len` | 在启动时调高，这是高基数 `choice` 的解法。选项共享它，所以给每个标签更多空间是保持它们可区分的唯一办法。每次调用都重新读取，所以设一次就够。 |
| `--max-len` | 总预算。调高它是 state 被截断的解法。 |
| `--concurrency` | 只有在你确知 Laya 不共享设备状态时才调高。默认 1 是正确性，不是谨慎。 |
| `--sidecar` | 让 `laya-mcp mcp` 指向一个正在运行的 `serve`。强烈推荐：harness 每个会话生成一个 stdio server，在每个里面托管模型要为每个会话付一次加载成本。 |

---

## 本包不修复的东西

上游自己的数字值得重复，因为一个暗示相反的集成层是在骗你。

* **基础 checkpoint 在有类型决策上零样本接近随机**——英文 0.362，对 **0.461 的多数类基线**。猜最常见的答案能赢过模型。
* **`score` 是最弱的 primitive。** 在一个五级有序任务上独立测得 35%，对比 Jev 的 70%。
* **标定需要标注数据。** 原始 ECE 英文 0.466、多语言 0.314，拟合后改善到 0.081 与 0.106。温度是编不出来的；本包不会假装可以。若拟合撞到搜索网格的边缘，它会作为「界」记入 `saturated_buckets`，而不是当作温度返回——因为界描述的是样本，不是模型。
* **位置偏置是真实存在的。** 一次公开的 fixture 运行在 50 道多选题里 46 次选了「A」。
* **超过约 20 个选项后准确率下滑**，据作者本人。

标定让概率*诚实*；它不能让模型*正确*。如果你的任务上准确率不在，就在自己的领域数据上拟合，或者不要部署。

---

## 验证

```bash
python tests/smoke_pure.py         # 85 项：校验、规划、标定、错误
python tests/install_harnesses.py  # 38 项：每一种 harness 方言，在临时目录里
python tests/mcp_protocol.py       # 连上 sidecar 时 25 项（离线 20 项）：真实的 MCP 握手与真实的工具调用
python tests/stdio_latency.py      # 握手 <5 秒、工具列表瞬时、工具调用有返回
python tests/language_probe.py     # 每个 checkpoint 在各语言上实际能做什么
laya-mcp doctor                    # 装了什么，以及 GPU 真正能做什么
```

三个套件共 148 项检查，各自覆盖其它套件够不到的一层。`stdio_latency.py` 与 `language_probe.py` 需要模型，是测量而非断言，因此手工运行，其数字在上文被引用。

`smoke_pure.py` 不需要 torch、模型、网络或 harness 配置。`install_harnesses.py` 把每个 harness 重定向到临时目录，因为 `~/.claude.json` 是一个装着历史与逐项目状态的大共享文件，一个覆盖它的测试会比它能抓到的任何 bug 更糟。

`mcp_protocol.py` 是最重要的一个，也是缺失最久的一个。它完全按 harness 的方式生成服务器（`python -m laya_mcp mcp`），用官方 SDK 做真实的 `initialize` 握手、列出工具并调用它们。两个真实缺陷逃过了其它套件，只在这里被抓到：mcp 1.30 里的 `FastMCP` 不接受 `version` 参数，导致服务器根本无法启动；以及 token 预算警告写进了 DSH 插件的工具描述里却从没写进本服务器的，于是走 MCP 的客户端无从知道过大的 state 会被从末尾截断。

随后，接受度是通过让 harness 解析**并连接到**本工具写出的文件来验证的——这是唯一能区分「文件被写了」与「文件被接受了」的测试：

| harness | 方式 | 结果 |
|---|---|---|
| opencode | `opencode mcp list` | ✓ connected |
| claude | `claude mcp list` | √ Connected |
| codex | `codex mcp list --json` | `enabled`、`"type": "stdio"`、参数正确——`auth_status: unsupported` 不是缺陷，本地 stdio server 不需要认证 |
| OpenClaw | `openclaw mcp list --json` | 报告出服务器与 stdio 传输 |
| Hermes | `hermes mcp list` | ✓ enabled，且 `hermes mcp test laya` 能连上并发现全部 5 个工具 |
| `pi` | — | 无原生 MCP 支持；`install` 会检测到并说明 |

这个安装器支持的每一个 harness，现在都由它自己的工具确认过了。这是唯一能区分「文件被写了」与「文件被接受了」的检查，而它证明了自己的价值：Windows 上 Hermes 从 `%LOCALAPPDATA%\hermes` 读配置，不是 `~/.hermes`——安装器一直在报成功，同时写进一个没人读的文件。两个缺陷掩盖了它：Hermes 被标为不可验证，而在 Windows 上这些 lister 一个都启动不了，因为 npm 给它们的都是 `.cmd` 垫片，`CreateProcess` 拒绝执行。

这张表正是 `stdio_latency.py` 存在的原因。上表中每个 harness 都只给 MCP server 30 秒完成 `initialize`，而在加载 checkpoint 之后才应答握手需要 19 秒（无争用）、275 秒（有另一个模型占着 GPU）——于是它们全都对一个解析得完全正确的配置报「Failed to connect」。现在加载跑在握手之后。

## 许可证

Apache-2.0。Laya 由 Convai Innovations 以 Apache-2.0 发布。这是一个独立集成，与该上游项目无关联，也未获其背书。
