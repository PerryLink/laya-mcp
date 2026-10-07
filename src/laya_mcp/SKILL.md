---
name: laya
description: Ask a local Laya model for typed noul, choice and score judgments and branch on the probabilities. Use when the next step is a decision over evidence you already have and the answer should be a number or a label rather than prose.
when-to-use: The current step is a judgment over text you already hold, and what follows depends on a typed answer. Skip it when the user needs generated text, a patch, or a shell command - Laya does not write, and a decision model asked to write produces nothing useful.
---

# Ask Laya

Laya is a non-autoregressive "System 1" decision model. It does not chat. You send
a `state` and typed `questions`; you get probabilities back in a single forward
pass. Everything runs locally: the model is a checkpoint on this machine and no
part of your state leaves it.

Call the `laya_ask` tool for a batch. `laya_noul`, `laya_choice` and `laya_score`
are one-question conveniences. `laya_plan` costs nothing and tells you what would
be cut before you spend a forward pass.

## When to use it

Use Laya where you would otherwise prompt an LLM to "decide" and then parse its
prose.

- Is this change in scope?
- Which team should own this ticket?
- How severe is this failure on a rubric you define?

Do not use it to generate a patch, a commit message, or an explanation. Keep
those with the session model.

## Question types

| Type | Use | Returns |
| --- | --- | --- |
| `noul` | Does this hold? | `noul` = P(true) in 0..1, plus a `no`/`uncertain`/`yes` band |
| `choice` | One option from a set you name | `choice`, `probabilities` |
| `score` | A degree on ordered levels you write | `score`, `legend`, `probabilities` |

Write the whole meaning into `instructions`. Question ids are for you.

## Always supply the option text

This is the one rule that decides whether an answer means anything.

A `noul` with no `boundary` renders its options as the fixed pair
`false: ...` / `true: ...`, and the checkpoint answers "false" to essentially
every one of them - measured at 40 of 40 items, in both English and Chinese,
scoring exactly chance. Give it the boundary:

```json
{
  "type": "noul",
  "instructions": "Does the user threaten to cancel?",
  "boundary": {
    "true": "the user says they will cancel or not renew",
    "false": "no such threat anywhere in the message"
  }
}
```

With a boundary the same forty items score 1.000 and 0.975. The difference is not
subtle; it is the difference between a decision and a constant.

For `choice`, two bare labels are often indistinguishable to the model - the
description is what separates them. Include a no-match option when nothing may
fit. Accuracy falls off sharply above roughly twenty options, and past that the
option text is silently shortened.

## Look before you leap

`laya_plan` runs the same arithmetic as an answer but no model, and reports what
would be cut. Use it when the state is long or the option list is wide.

Laya truncates the state **from the end** and says nothing in the answer. For a
contract, a log, or an email thread, the end is usually where the answer was.
Read `truncated`, `budget_summary` and `warnings` on every response, or pass
`strict: true` to have an oversized request refused instead.

## How to read the numbers

- **`confidence` is not accuracy.** It is a concentration statistic over the
  distribution: low when probability is spread out even when the top option is
  right, and high on a confident wrong answer. Branch on the probability, not on
  this.
- **A `noul` near 0.5 is undecided, not a weak yes.** The `band` exists because a
  calibrated probability is not a decision.
- **The base checkpoints are near chance zero-shot on typed decisions** - 0.362
  against a 0.461 majority-class baseline for English. Calibration can make a
  probability honest; it cannot make the model right. Fit a temperature on your
  own labelled examples before you rely on the numbers, and do not deploy it for
  a task where the accuracy is not there.
- **Language matters.** `english` and `typed-decisions` are English encoders.
  Ask in Chinese without `lang: "zh"` and you get the English checkpoint, with a
  warning on the response. Pass `lang` for anything that is not English.

## How to call

```json
{
  "state": { "subject": "Duplicate charge", "body": "Billed twice. Refund or we cancel." },
  "questions": {
    "churn": {
      "type": "noul",
      "instructions": "Does the user threaten to cancel?",
      "boundary": {
        "true": "an explicit threat to cancel or not renew",
        "false": "no threat, however annoyed the tone"
      }
    },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages",
        "other": "none of the above"
      }
    }
  }
}
```

Several questions in one call share one forward pass. Prefer one call over a
sequence when the questions do not depend on each other. Weighing independent
factors against each other is your job, not Laya's - ask separate questions and
combine them yourself.

## When it fails

- `sidecar_unreachable` - no model process is answering. Start one with
  `laya-mcp serve`, or set the plugin's `lifecycle: spawn` and a `spawnCommand`
  so the harness starts it for you.
- `invalid_question` - the code names the offending question id and a hint.
- `state_truncated` - only in `strict` mode; shorten the state or raise
  `max_len` at startup.

Do not invent an answer when a call fails. Report it.
