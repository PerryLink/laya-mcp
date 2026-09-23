"""Regression tests for the token-budget preflight: no torch, no model, no network.

Everything asserted here is a property the project measured and then got wrong
once, so a failure means a defect has come back. Run it with the package on
sys.path:

    python tests/preflight_contract.py

WHY EACH CHECK EXISTS

A. `fits` must be a boolean. A client reported
   `tool "laya_plan" returned invalid output: "value.fits" must be a boolean`.
   Auditing this package found it *cannot* emit a non-boolean -- `fits` has one
   assignment, `fits=not any_truncation` -- so the violation is raised by whatever
   validated the result. That makes the contract easy to break by accident and
   invisible from here, which is exactly when a test should hold it.

B. `_looks_non_latin` must answer the same way regardless of where the CJK sits
   in the text. The old implementation averaged over `text[:4000]` only, so on a
   document that opens with an English title and continues in Chinese the flag
   flipped True -> False -> True as the prefix grew, and a wholly Chinese state
   between 239 and 830 characters was measured with the *Latin* ratio: budget
   under-reserved by 2.5x, evidence silently cut.

C. ASCII-escaped CJK must classify as CJK. `json.dumps` defaults to
   `ensure_ascii=True`, which leaves `u`/`e`/`d`/`f` as the only letters, so the
   old test measured a Latin fraction of 1.000 on wholly Chinese content and
   under-reserved by 2.4x.

D. A supplied tokenizer must turn the state figure from an estimate into a count.
   `plan_questions` has always accepted one and no production caller passed it, so
   `exact` was `false` on every response and the character estimate alone decided
   `fits` -- including on JSON, code and CJK, where that estimate UNDER-reserves.
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
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}  {detail}")
        FAILURES.append(label)


def main() -> int:
    import json

    from laya_mcp.capability import Capability
    from laya_mcp.planning import _escaped_non_latin_ratio, _looks_non_latin, plan_questions
    from laya_mcp.protocol import Question

    cap = Capability(
        checkpoint="english", repo="test", subfolder=None, encoder=None,
        device="cpu", requested_device=None, degraded=False,
        max_len=512, head_max_len=192, head_layers=None,
        temperature=(1.0, 1.0, 1.0), fitted_temperature_buckets=0, amp_dtype=None,
    )
    noul = Question(type="noul", instructions="Is the claim true?", id="q1")

    # ---- A. fits is a boolean, at every verdict the planner can reach ---------
    for label, state in (
        ("tiny", "hello"),
        ("fits", "word " * 200),
        ("does not fit", "word " * 4000),
    ):
        payload = plan_questions(cap, state, {"q1": noul}).to_dict()
        check(f"A fits is a bool ({label})",
              type(payload["fits"]) is bool,
              f"got {type(payload['fits']).__name__} = {payload['fits']!r}")
    # And in the strict-refusal path the same field is what a caller reads.
    big = plan_questions(cap, "word " * 4000, {"q1": noul})
    check("A strict gate reads a bool", big.fits is False or big.fits is True,
          f"fits={big.fits!r}")

    # ---- B. classification does not depend on where the CJK sits --------------
    zh_body = "这是中文内容，用来检验语言分类是否正确。" * 40
    for prefix in ("", "English title. ", "E" * 200 + ". ", "A short English abstract. " * 8):
        text = prefix + zh_body
        check(f"B Chinese with {len(prefix):>4}-char Latin prefix is dense",
              _looks_non_latin(text) is True,
              f"prefix={prefix[:24]!r}")

    # The exact regression: a wholly Chinese state in the bad band.
    for n in (300, 500, 830, 1000):
        check(f"B pure Chinese at {n} chars is dense",
              _looks_non_latin(zh_body[:n]) is True)

    # A genuinely English document must NOT be classified dense, or the estimator
    # over-reserves by 4x on the common case.
    english = ("Handing the judgment work in a long-horizon task to a cheap, "
               "specialised judge is a division of labour usually sold on the "
               "grounds that it saves money. " * 20)
    check("B English prose is not dense", _looks_non_latin(english) is False)

    # ---- C. ASCII-escaped CJK is CJK -----------------------------------------
    escaped = json.dumps(zh_body)                       # ensure_ascii=True default
    check("C escape ratio is detected",
          (_escaped_non_latin_ratio(escaped) or 0) > 0.5,
          f"ratio={_escaped_non_latin_ratio(escaped)}")
    check("C escaped CJK is dense", _looks_non_latin(escaped) is True)

    # Ordinary JSON with ASCII keys must not be dragged into the dense branch.
    ascii_json = json.dumps(
        [{"arm": "FULL-001", "n": 10, "accuracy": 0.5} for _ in range(40)]
    )
    check("C ASCII JSON is not dense", _looks_non_latin(ascii_json) is False)

    # ---- D. a supplied tokenizer makes the count exact ------------------------
    class FakeTokenizer:
        """Stands in for the checkpoint's tokenizer: 8 chars per token."""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(max(1, len(text) // 8)))}

    state = "x" * 400
    estimated = plan_questions(cap, state, {"q1": noul})
    exact = plan_questions(cap, state, {"q1": noul}, tokenizer=FakeTokenizer())
    check("D no tokenizer means an estimate", estimated.exact is False)
    check("D a tokenizer means an exact count", exact.exact is True,
          f"exact={exact.exact} warning={exact.warnings}")
    check("D the exact figure comes from the tokenizer",
          exact.state_tokens_estimated == 50,
          f"got {exact.state_tokens_estimated}, expected 400/8 = 50")
    check("D the estimate differs from the count, as documented",
          estimated.state_tokens_estimated != exact.state_tokens_estimated,
          f"estimate={estimated.state_tokens_estimated} exact={exact.state_tokens_estimated}")

    # A tokenizer that cannot count must fall back LOUDLY, not silently.
    class BrokenTokenizer:
        def __call__(self, text, add_special_tokens=False):
            raise TypeError("not a transformers tokenizer")

    broken = plan_questions(cap, state, {"q1": noul}, tokenizer=BrokenTokenizer())
    check("D an unusable tokenizer falls back and says so",
          broken.exact is False and any("estimate" in w for w in broken.warnings),
          f"exact={broken.exact} warnings={broken.warnings}")

    # ---- E. BOTH tokenizer protocols must be accepted -------------------------
    #
    # This is the difference between a fix and the appearance of one. `planning`
    # originally supported only the transformers call shape; a bare
    # `tokenizers.Tokenizer` -- which is what `tokenizer.json` loads into, and
    # what this project's own measurements use -- raised TypeError, was caught,
    # warned, and answered with the character estimate. The budget stayed wrong
    # while the code looked repaired.
    class BareTokenizersStyle:
        """The `tokenizers` library's shape: .encode(...).ids, not callable."""

        class _Encoded:
            def __init__(self, n): self.ids = list(range(n))

        def encode(self, text, add_special_tokens=False):
            return BareTokenizersStyle._Encoded(max(1, len(text) // 8))

    bare = plan_questions(cap, state, {"q1": noul}, tokenizer=BareTokenizersStyle())
    check("E the bare tokenizers.Tokenizer protocol is accepted",
          bare.exact is True,
          f"exact={bare.exact} warnings={bare.warnings}")
    check("E and it produces the same count as the transformers shape",
          bare.state_tokens_estimated == exact.state_tokens_estimated,
          f"bare={bare.state_tokens_estimated} transformers={exact.state_tokens_estimated}")

    # An object exposing neither protocol must still be refused rather than
    # crashing the planner.
    neither = plan_questions(cap, state, {"q1": noul}, tokenizer=object())
    check("E an object with neither protocol falls back rather than raising",
          neither.exact is False and neither.state_tokens_estimated > 0)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} failure(s) of {CHECKS} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"RESULT: all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
