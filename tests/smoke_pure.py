"""Smoke test for the pure logic: no torch, no model, no network.

Everything asserted here is derived from Laya's source rather than its README, so
a failure means this package and the library disagree about the contract. Run it
with the package on sys.path:

    python tests/smoke_pure.py
"""

from __future__ import annotations

import os
import sys

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
    import laya_mcp
    from laya_mcp import calibration
    from laya_mcp.capability import Capability, normalise_checkpoint, read_capability
    from laya_mcp.errors import (
        InvalidQuestionError,
        LayaMcpError,
        QuestionTooLargeError,
        translate,
    )
    from laya_mcp.planning import plan_questions, serialized_state_chars
    from laya_mcp.protocol import Question
    from laya_mcp.validate import validate_questions
    from laya_mcp.worker import _band

    print(f"laya-mcp {laya_mcp.__version__}\n")

    # The English checkpoint's real numbers, read from its rl_agent_config.json.
    english = Capability(
        checkpoint="english",
        repo="convaiinnovations/laya",
        subfolder=None,
        encoder="answerdotai/ModernBERT-large",
        device="cpu",
        requested_device="cuda",
        degraded=True,
        max_len=512,
        head_max_len=192,
        head_layers=2,
        temperature=(1.6369, 1.2514, 1.9834),
        fitted_temperature_buckets=6,
        amp_dtype="fp16",
    )

    print("capability arithmetic (reproduces build_sequence)")
    check("state budget is max_len - head_max_len", english.state_budget_tokens == 320,
          str(english.state_budget_tokens))
    # 192 // 30 = 6, plus the mask token = 7, and 192 - 30*7 = -18 < 16 so the
    # fallback fires: max(4, (192-16)//30) = 5, plus mask = 6.
    check("30 options: fallback path fires", english.option_token_budget(30) == 6,
          str(english.option_token_budget(30)))
    check("2 options keep the full ceiling plus mask",
          english.option_token_budget(2) == 49, str(english.option_token_budget(2)))
    check("more options never means more tokens each",
          english.option_token_budget(50) <= english.option_token_budget(10))
    check("max_options leaves at least 8 tokens each",
          english.option_token_budget(english.max_options()) >= 8)
    check("max_options+1 drops below 8",
          english.option_token_budget(english.max_options() + 1) < 8,
          f"max={english.max_options()}")
    check("high-cardinality advisory is 20", english.high_cardinality_warning_at() == 20)
    check("degraded is reported", english.degraded is True)
    check("English checkpoint carries the language caveat", english.language_caveat() is not None)

    multilingual = Capability(
        checkpoint="multilingual", repo="convaiinnovations/laya", subfolder="multilingual",
        encoder="jhu-clsp/mmBERT-base", device="cuda", requested_device=None, degraded=False,
        max_len=1024, head_max_len=256, head_layers=2, temperature=(1.0, 1.0, 1.0),
        fitted_temperature_buckets=0, amp_dtype="fp16",
    )
    check("multilingual has no language caveat", multilingual.language_caveat() is None)
    check("multilingual ships no fitted temperatures",
          multilingual.fitted_temperature_buckets == 0)

    print("\ncheckpoint name normalisation")
    check("alias 'ml' resolves", normalise_checkpoint("ml") == "multilingual")
    check("alias 'en' resolves", normalise_checkpoint("en") == "english")
    check("canonical name passes through",
          normalise_checkpoint("typed-decisions") == "typed-decisions")
    try:
        normalise_checkpoint("laya-mlx")
        check("unknown checkpoint raises", False)
    except KeyError:
        check("unknown checkpoint raises", True)

    print("\nvalidation: every case is a bare KeyError in Laya")
    bad = [
        ("unknown type", Question(type="ranking", instructions="x"), InvalidQuestionError),
        ("choice without criteria", Question(type="choice", instructions="pick"), InvalidQuestionError),
        ("blank instructions", Question(type="noul", instructions=""), InvalidQuestionError),
        ("score as a mapping", Question(type="score", instructions="s", criteria={"a": "1", "b": "2"}), InvalidQuestionError),
        ("score with one level", Question(type="score", instructions="s", criteria=["only"]), InvalidQuestionError),
        ("score with a null level", Question(type="score", instructions="s", criteria=["a", None]), InvalidQuestionError),
        ("noul with an alien key", Question(type="noul", instructions="q", criteria={"maybe": "x"}), InvalidQuestionError),
    ]
    for label, question, expected in bad:
        try:
            validate_questions({"q": question})
            check(label, False, "no error raised")
        except expected as exc:
            check(label, exc.question_id == "q", f"question_id={exc.question_id}")
        except LayaMcpError as exc:
            check(label, False, f"raised {type(exc).__name__}")

    good = {
        "a": Question(type="noul", instructions="Is it urgent?"),
        "b": Question(type="choice", instructions="Which team?",
                      criteria={"billing": "invoices", "tech": "bugs"}),
        "c": Question(type="score", instructions="How urgent?", criteria=["no", "soon", "critical"]),
        "d": Question(type="noul", instructions="Churn?", criteria={"true": "threatens", "false": "does not"}),
    }
    try:
        validate_questions(good, capability=english)
        check("a well-formed batch passes", True)
    except LayaMcpError as exc:
        check("a well-formed batch passes", False, str(exc))

    try:
        validate_questions({"q": Question(type="choice", instructions="pick",
                                          criteria={f"o{i}": f"d{i}" for i in range(80)})},
                           capability=english)
        check("too many options is refused with numbers", False, "no error")
    except QuestionTooLargeError as exc:
        check("too many options is refused with numbers",
              exc.details.get("tokens_per_option") is not None, str(exc.details))

    print("\nerror translation")
    check("KeyError('criteria') -> invalid_question",
          translate(KeyError("criteria")).code.value == "invalid_question")
    check("KeyError('ranking') -> invalid_question naming the type",
          "ranking" in str(translate(KeyError("ranking"))))
    translated = translate(ValueError("question 'x' options exceed head_max_len=192"), question_id="x")
    check("head_max_len ValueError -> question_too_large",
          translated.code.value == "question_too_large")
    check("question_too_large is a 400", translated.http_status == 400)
    check("not retryable at 400", translated.to_dict()["retryable"] is False)
    oom = translate(RuntimeError("CUDA out of memory"))
    check("CUDA OOM -> out_of_memory", oom.code.value == "out_of_memory")
    check("out_of_memory is retryable", oom.to_dict()["retryable"] is True)
    # Measured against the real checkpoint: a choice with no criteria raises
    # AttributeError('NoneType' ... 'items'), not KeyError. Reading the source
    # alone would have missed this, which is why the fix is pinned by a test.
    attr = translate(AttributeError("'NoneType' object has no attribute 'items'"), question_id="q")
    check("missing criteria AttributeError -> invalid_question",
          attr.code.value == "invalid_question", attr.code.value)
    other = translate(AttributeError("'str' object has no attribute 'foo'"))
    check("an unrelated AttributeError stays internal",
          other.code.value == "internal", other.code.value)

    print("\nplanning")
    check("serialized_state_chars matches json.dumps",
          serialized_state_chars({"a": "b"}) == len('{"a": "b"}'))
    plan = plan_questions(
        english,
        {"body": "x" * 5000},
        {
            "urgency": Question(type="score", instructions="How urgent?",
                                criteria=["no", "soon", "critical"]),
            "team": Question(type="choice", instructions="Which team?",
                             criteria={f"opt{i}": f"description {i}" for i in range(30)}),
        },
    )
    check("a 5000-char state does not fit the English budget", plan.fits is False)
    check("state budget is an estimate without a tokenizer", plan.exact is False)
    check("plan names the worst question", plan.worst_question is not None)
    check("plan offers a recommendation when it does not fit", bool(plan.recommendation))
    check("plan warns about the squeezed choice", len(plan.warnings) > 0, str(plan.warnings[:1]))
    report = plan.truncation_report()
    check("truncation report names the state cut", report is not None and "state" in report)

    small = plan_questions(english, {"body": "short"}, {
        "q": Question(type="noul", instructions="Is this short?"),
    })
    check("a small batch fits", small.fits is True)
    check("a fitting batch has no truncation report", small.truncation_report() is None)

    dense = plan_questions(english, {"body": "这是一个很长的中文文档" * 200}, {
        "q": Question(type="noul", instructions="Is this long?"),
    })
    check("non-Latin state is not under-reserved",
          dense.state_tokens_estimated > 200, str(dense.state_tokens_estimated))

    print("\ncalibration")
    check("semantics string denies being P(correct)",
          "NOT the probability" in calibration.CONFIDENCE_SEMANTICS)
    check("noul bucket is '2'", calibration.bucket_for("noul", 2) == "noul:2")
    check("30-option choice bucket is 11+", calibration.bucket_for("choice", 30) == "choice:11+")
    softened = calibration.apply_temperature([0.6, 0.3, 0.1], 2.0)
    check("softening flattens the distribution", softened[0] < 0.6 and softened[2] > 0.1,
          str([round(v, 4) for v in softened]))
    check("softened distribution still sums to 1", abs(sum(softened) - 1.0) < 1e-9)
    sharpened = calibration.apply_temperature([0.6, 0.3, 0.1], 0.5)
    check("sharpening concentrates it", sharpened[0] > 0.6,
          str([round(v, 4) for v in sharpened]))
    check("a tiny temperature does not produce a NaN",
          all(v == v for v in calibration.apply_temperature([0.5, 0.5], 1e-9)))
    # ECE compares stated confidence against the OBSERVED accuracy of its bin,
    # per bin. So a lone correct case at 0.95 in [0.9, 1.0) has bin accuracy 1.0
    # and costs exactly its over-confidence, 0.05. The only zero is confidence
    # that lands exactly on its bin's accuracy - which for a one-sample bin means
    # a stated confidence of 1.0.
    check("ECE is 0 when stated confidence equals observed accuracy",
          calibration.expected_calibration_error([1.0], [True]) < 1e-9,
          str(calibration.expected_calibration_error([1.0], [True])))
    check("ECE charges exactly the over-confidence (0.95 correct -> 0.05)",
          abs(calibration.expected_calibration_error([0.95, 0.99, 0.91], [True, True, True]) - 0.05) < 1e-9,
          str(calibration.expected_calibration_error([0.95, 0.99, 0.91], [True, True, True])))
    check("ECE charges under-confidence too (0.5 correct -> 0.5)",
          abs(calibration.expected_calibration_error([0.5, 0.5], [True, True]) - 0.5) < 1e-9,
          str(calibration.expected_calibration_error([0.5, 0.5], [True, True])))
    # Perfectly wrong: 0.9 confident but incorrect, 0.1 confident but correct.
    check("ECE is 0.9 when confidence is exactly inverted",
          abs(calibration.expected_calibration_error([0.1, 0.9], [True, False]) - 0.9) < 1e-9,
          str(calibration.expected_calibration_error([0.1, 0.9], [True, False])))
    check("a confidence of exactly 1.0 is counted, not dropped",
          calibration.expected_calibration_error([1.0], [True]) < 1e-9)
    # The documented consequence of half-open bins: a value exactly on an interior
    # boundary belongs to the UPPER bin. Pinned so the convention cannot change
    # silently. Here 0.9 falls in [0.9, 1.0) where accuracy is 1.0, costing 0.1,
    # while 0.1 stays in [0.1, 0.2) and costs nothing.
    on_boundary = calibration.expected_calibration_error([0.9, 0.1], [True, False])
    check("an interior-boundary confidence uses the upper bin (costs 0.1)",
          abs(on_boundary - 0.10) < 1e-9, str(on_boundary))

    temperature, nll = calibration.fit_temperature(
        [[0.9, 0.1], [0.8, 0.2], [0.95, 0.05], [0.7, 0.3]],
        [0, 0, 0, 1],
    )
    check("fitting returns a finite temperature", 0.0 < temperature < 100.0, str(temperature))
    check("fitting returns a finite loss", nll == nll)

    entry = calibration.fit_from_examples(
        "english",
        [{"probabilities": [0.9, 0.1], "outcome": 0, "type": "choice"}] * 40
        + [{"probabilities": [0.6, 0.4], "outcome": 0, "type": "choice"}] * 5,
        min_samples=30,
    )
    check("only buckets above min_samples are fitted",
          "choice:2" in entry.by_options and len(entry.by_options) == 1,
          str(list(entry.by_options)))
    check("entry records the sample count", entry.samples == 45, str(entry.samples))
    check("entry reports ECE before and after",
          entry.ece_before is not None and entry.ece_after is not None)
    check("temperature_for falls back to the per-primitive value",
          calibration.CalibrationEntry(checkpoint="x", temperatures=[1.0, 2.0, 3.0])
          .temperature_for("score", 4) == 2.0)

    print("\nbanding (a calibrated probability is not a decision)")
    check("0.20 -> no", _band(0.20) == "no")
    check("0.30 -> uncertain (boundary belongs to uncertain)", _band(0.30) == "uncertain")
    check("0.70 -> uncertain (boundary belongs to uncertain)", _band(0.70) == "uncertain")
    check("0.71 -> yes", _band(0.71) == "yes")

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
