"""Measure what the checkpoints can actually do, in the language being asked.

Written after a first attempt produced a wrong answer that looked like a right
one. That attempt asked a `noul` - "Is this review positive?" - and reported
0.550 on Chinese against a 0.500 baseline, which reads as "the multilingual
checkpoint is at chance in Chinese". Re-asking one sentence four ways showed the
measurement was of the wrong thing:

    noul  "Is this review positive?"   -> 0.0000   (says: no)
    noul  "Is this review NEGATIVE?"   -> 0.0002   (says: no)
    choice {positive, negative}        -> positive at 0.9268   <- correct
    score  5 levels                    -> 2.10 (neutral-positive)

`noul` returns approximately zero for every review-shaped sentence in this set,
whichever way the question is put and whether or not criteria are supplied, while
`choice` reads the same sentence correctly. Measured on the English set, all
forty items came back below 0.0001 - a constant, not a judgment.

So this file measures both framings and both languages. The noul/choice gap is
not a detail of the harness; it is the finding, and it is invisible if you only
ever use the primitive that happens to work.

**What this is.** Forty balanced items per language (twenty positive reviews,
twenty negative), so the majority baseline is exactly 0.500. Forty items puts the
standard error near 0.08: this resolves "chance" from "working" and cannot
resolve small differences. It is a probe, not a benchmark.

**What it is not.** Not a claim about the checkpoints in general. One task, one
domain, one phrasing family. A model that scores 0.80 here may be far worse on
the thing you actually want to classify.

Run:  python tests/language_probe.py --sidecar http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence

ZH_POSITIVE = [
    "这个产品质量很好，用了半年一点问题都没有。",
    "客服响应很快，问题当天就解决了，非常满意。",
    "物流很快，包装也很仔细，东西完好无损。",
    "性价比很高，比同价位的其他品牌好太多。",
    "第二次回购了，一如既往地好用。",
    "界面简洁，上手很容易，新手也能很快学会。",
    "做工精致，细节处理得非常好。",
    "电池续航超出预期，一天重度使用还有余量。",
    "安装很简单，说明书写得很清楚。",
    "味道很正宗，家里人都很喜欢。",
    "退换货流程很顺畅，没有任何扯皮。",
    "运行安静，几乎听不到风扇声音。",
    "屏幕色彩鲜艳，看视频很舒服。",
    "快递小哥态度很好，还帮忙搬上楼。",
    "用了三个月，性能依然稳定，没有卡顿。",
    "尺寸和描述完全一致，放在桌上刚刚好。",
    "售后服务很到位，主动打电话回访。",
    "材质摸着很舒服，没有异味。",
    "功能齐全，该有的都有，超出预期。",
    "价格实惠，活动期间买的更划算。",
]

ZH_NEGATIVE = [
    "收到货就是坏的，联系客服也没人理。",
    "用了两天就出现故障，质量太差了。",
    "物流慢得离谱，等了半个月才到。",
    "实物和图片差距很大，颜色完全不一样。",
    "客服态度恶劣，问题拖了一周都没解决。",
    "噪音特别大，晚上根本没法用。",
    "电池很不耐用，半天就没电了。",
    "做工粗糙，边角还有毛刺。",
    "说明书完全看不懂，安装折腾了一下午。",
    "味道很奇怪，吃了一口就扔了。",
    "申请退货被拒绝，说是人为损坏。",
    "价格虚高，完全不值这个钱。",
    "屏幕有明显的坏点，看着很别扭。",
    "快递员直接把包裹扔在门口，也没通知。",
    "用了一个月就开始卡顿，越用越慢。",
    "尺寸比描述小很多，根本装不下。",
    "售后推诿责任，来回踢皮球。",
    "塑料味很重，通风一周都散不掉。",
    "功能阉割严重，宣传的功能一半都没有。",
    "刚买就降价了，客服还不给补差价。",
]

EN_POSITIVE = [
    "This product is excellent quality - six months in and not a single problem.",
    "Support replied fast and fixed it the same day. Very satisfied.",
    "Shipping was quick, the packaging was careful, and it arrived undamaged.",
    "Great value for money, much better than other brands at this price.",
    "Bought it a second time. Just as good as the first.",
    "The interface is clean and easy to pick up, even for a beginner.",
    "Beautifully made, the details are handled very well.",
    "Battery life beat my expectations - still going after a heavy day.",
    "Installation was simple and the manual was clear.",
    "The flavour is authentic and the whole family likes it.",
    "Returns were painless, no arguing at all.",
    "It runs quietly - you can barely hear the fan.",
    "The screen colours are vivid and it is comfortable to watch on.",
    "The courier was friendly and even carried it upstairs.",
    "Three months in, performance is still stable with no stutter.",
    "The size matches the description exactly and fits the desk perfectly.",
    "After-sales was proactive and called to follow up.",
    "The material feels good and there is no odd smell.",
    "Fully featured - it has everything and more than I expected.",
    "Good price, and even better bought during the sale.",
]

EN_NEGATIVE = [
    "It arrived broken and nobody answers when I contact support.",
    "It failed after two days. The quality is terrible.",
    "Shipping was absurdly slow - it took half a month to arrive.",
    "The item looks nothing like the photos, the colour is completely different.",
    "Support was rude and left the problem unresolved for a week.",
    "It is extremely loud, unusable at night.",
    "The battery does not last - half a day and it is dead.",
    "Roughly made, with burrs on the edges.",
    "The manual is incomprehensible; installation took me a whole afternoon.",
    "The flavour is strange. One bite and I threw it away.",
    "My return was rejected as user damage.",
    "Overpriced. Not worth the money at all.",
    "The screen has obvious dead pixels and it is unpleasant to look at.",
    "The courier dumped the parcel at the door without telling me.",
    "After a month it started stuttering and gets slower every day.",
    "Much smaller than described, it does not fit at all.",
    "After-sales passes the blame around and nothing gets resolved.",
    "A strong plastic smell that a week of airing did not shift.",
    "Heavily stripped down - half the advertised features are missing.",
    "It dropped in price right after I bought it and support refused the difference.",
]

#: (language, positive items, negative items, noul question, choice question,
#:  choice criteria, the key meaning "positive", the lang to send, expected model)
CASES = [
    (
        "english / english",
        EN_POSITIVE, EN_NEGATIVE,
        "Is this review positive?",
        "Is this review positive or negative?",
        {"positive": "a positive review", "negative": "a negative review"},
        "positive", None, "english",
    ),
    (
        "chinese / multilingual",
        ZH_POSITIVE, ZH_NEGATIVE,
        "这条评论是正面的吗？",
        "这条评论是正面的还是负面的？",
        {"正面": "评论表达的是正面评价", "负面": "评论表达的是负面评价"},
        "正面", "zh", "multilingual",
    ),
]


def post(sidecar: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
    request = urllib.request.Request(
        f"{sidecar.rstrip('/')}/ask",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _score(
    sidecar: str, text: str, question: Mapping[str, Any], lang: Optional[str]
) -> tuple[Optional[str], Optional[str], float]:
    """Return (chosen label, model, P(the choice it made)."""
    body: dict[str, Any] = {"state": {"body": text}, "questions": {"q": question}}
    if lang:
        body["lang"] = lang
    payload = post(sidecar, body)
    answer = (payload.get("answers") or {}).get("q") or {}
    model = payload.get("model")
    if answer.get("type") == "noul":
        p_true = float(answer.get("noul") or 0.0)
        return ("true" if p_true >= 0.5 else "false"), model, max(p_true, 1.0 - p_true)
    chosen = answer.get("choice")
    probs = answer.get("probabilities") or {}
    return chosen, model, float(probs.get(chosen, 0.0)) if chosen else 0.0


def measure(sidecar: str, case, framing: str) -> dict:
    label, pos, neg, noul_q, choice_q, criteria, positive, lang, expected = case
    if framing == "noul":
        question: dict[str, Any] = {"type": "noul", "instructions": noul_q}
    elif framing == "noul_with_criteria":
        # A noul as the caller would send it, with the option text supplied. The
        # sidecar carries this as a choice under neutral labels; asking it here
        # rather than as a raw choice is what measures the adapter instead of the
        # idea behind it.
        labels = list(criteria.keys())
        negative = next(key for key in labels if key != positive)
        question = {
            "type": "noul",
            "instructions": noul_q,
            "criteria": {"true": criteria[positive], "false": criteria[negative]},
        }
    elif framing == "noul_as_choice":
        # The same question, in the same words, asked as a two-option choice
        # whose options carry the meaning - and labelled `A`/`B` rather than
        # `true`/`false`, which is the whole point.
        #
        # `noul` renders its options as `false: ...` / `true: ...`, hardcoded
        # upstream, and the model answers "false" to essentially every noul
        # whatever the state says. Measured on one positive review, varying only
        # the option labels and holding the question and the sentence fixed:
        #
        #     keys positive/negative  -> positive   0.835   correct
        #     keys true/false         -> false      true=0.000  wrong
        #     keys yes/no             -> no         yes=0.022   wrong
        #     keys A/B                -> A          0.798   correct
        #     keys 1/2                -> 1          0.906   correct
        #
        # So it is neither the primitive nor the option order - `yes`/`no` chose
        # the second option and `true`/`false` the first. It is the literal label
        # token: a `no`/`false` prior that dominates as soon as one is present,
        # and a noul cannot avoid one because it always renders both.
        #
        # Neutral labels are the workaround available to a caller. This framing
        # exists to measure whether it holds up over forty balanced items.
        labels = list(criteria.keys())
        negative = next(key for key in labels if key != positive)
        question = {
            "type": "choice",
            "instructions": noul_q,
            "criteria": {"A": criteria[positive], "B": criteria[negative]},
        }
    else:
        question = {"type": "choice", "instructions": choice_q, "criteria": criteria}

    TRUE_KEY = {
        "noul": "true",
        "noul_with_criteria": "true",
        "noul_as_choice": "A",
        "choice": positive,
    }[framing]

    rows = [(t, True) for t in pos] + [(t, False) for t in neg]
    hits, models, logged = 0, set(), []
    for text, truth in rows:
        try:
            chosen, model, p = _score(sidecar, text, question, lang)
        except urllib.error.URLError as exc:
            print(f"  request failed: {exc}")
            return {"error": str(exc)}
        models.add(model)
        correct = (chosen == TRUE_KEY) == truth
        hits += correct
        logged.append((truth, p, correct, chosen))

    total = len(rows)
    accuracy = hits / total
    mean_p = sum(p for _, p, _, _ in logged) / total
    # How many items came back as the "no" answer, whichever key spells it.
    FALSE_KEYS = {
        "noul": {"false"},
        "noul_with_criteria": {"false"},
        "noul_as_choice": {"B"},
        "choice": set(),
    }[framing]
    near_zero = sum(1 for _, _, _, chosen in logged if chosen in FALSE_KEYS)
    return {
        "label": label,
        "framing": framing,
        "model": ", ".join(sorted(str(m) for m in models)),
        "accuracy": accuracy,
        "se": math.sqrt(0.25 / total),
        "mean_p": mean_p,
        "total": total,
        "said_no": near_zero,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", default="http://127.0.0.1:8787")
    parser.add_argument(
        "--framing",
        choices=["noul", "noul_with_criteria", "noul_as_choice", "choice", "all"],
        default="all",
    )
    args = parser.parse_args()

    framings = (
        ["noul", "noul_with_criteria", "noul_as_choice", "choice"]
        if args.framing == "all"
        else [args.framing]
    )
    results = []
    for case in CASES:
        for framing in framings:
            print(f"running {case[0]} as {framing} ...")
            result = measure(args.sidecar, case, framing)
            if "error" in result:
                return 2
            results.append(result)

    print(f"\n{'case':<24} {'framing':<8} {'model':<13} {'accuracy':>9} {'±se':>6} {'meanP':>7}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<24} {r['framing']:<8} {r['model']:<13} "
            f"{r['accuracy']:>9.3f} {r['se']:>6.3f} {r['mean_p']:>7.3f}"
        )

    print("\nmajority baseline is 0.500 for every row (balanced 20/20).")
    noul_rows = [r for r in results if r["framing"] in ("noul", "noul_with_criteria", "noul_as_choice")]
    for r in noul_rows:
        print(
            f"  {r['label']:<24} {r['framing']:<16} answered 'no' on "
            f"{r['said_no']:>2}/{r['total']} items, scored {r['accuracy']:.3f}"
        )
    print("\nIf a noul row sits near 0.500 with a high 'no' count while the matching")
    print("choice row scores well on the same sentences, the primitive - not the")
    print("language - is what fell over. That is the failure this file exists to catch.")
    print("`noul_as_choice` asks the identical question as a two-option choice; when")
    print("it scores well and `noul` does not, the type embedding is at fault and the")
    print("framing is the workaround.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
