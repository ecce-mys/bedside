"""표현 드릴 채점(11-22, Morning Words). 받아쓴 대답이 목표 표현을 말했는지 본다.

표준 라이브러리만 쓴다: 음성 서버(helpers/claude_live.py, uv 환경)도 그대로 import 한다.
받아쓰기 오차(철자·축약형·한 글자 차이)는 봐주고, 핵심 표현이 빠지면 틀린 걸로 본다.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_CONTR = [
    (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon't\b", "will not"), (r"\bshan't\b", "shall not"),
    (r"\blet's\b", "let us"), (r"n't\b", " not"), (r"'m\b", " am"), (r"'re\b", " are"), (r"'ve\b", " have"),
    (r"'ll\b", " will"), (r"'d\b", " would"), (r"\b(it|that|what|there|here|he|she|who|where)'s\b", r"\1 is"),
]

PASS_RE = re.compile(r"\b(pass|skip|next one|no idea|i don'?t know|dunno|give up)\b|모르|패스|넘어가", re.I)
STOP_RE = re.compile(r"\b(stop|let'?s stop|that'?s it for today|i'?m done|quit)\b|그만|끝내", re.I)
READY_RE = re.compile(r"\b(ready|okay|ok|yes|yeah|yep|sure|let'?s go|go ahead|start|let'?s do it|i'?m up)\b"
                      r"|레디|시작|응|좋아|오케이|그래", re.I)


def norm(s: str) -> str:
    s = (s or "").lower().replace("’", "'").replace("‘", "'")
    for pat, rep in _CONTR:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s).replace("'", "")
    return " ".join(s.split())


def tokens(s: str) -> list[str]:
    return norm(s).split()


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) <= 3 or len(b) <= 3:  # 짧은 낱말(to/do, a/i)은 정확히 같아야
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def has_phrase(ans: list[str], key: list[str], thr: float = 0.75, gap: int = 2) -> bool:
    """key 낱말들이 대답 안에 순서대로(사이에 낱말 gap개까지 끼어도) 있으면 참."""
    if not key:
        return True
    for start in range(len(ans)):
        if _sim(ans[start], key[0]) < thr:
            continue
        i, ok = start, True
        for k in key[1:]:
            j = next((j for j in range(i + 1, min(len(ans), i + 2 + gap)) if _sim(ans[j], k) >= thr), None)
            if j is None:
                ok = False
                break
            i = j
        if ok:
            return True
    return False


def grade(answer: str, en: str, keys: list[str] | None) -> bool:
    """핵심 표현이 다 있으면 정답. 핵심이 없거나 못 찾으면 문장 전체가 아주 비슷할 때만 정답."""
    a = tokens(answer)
    if not a:
        return False
    if keys and all(has_phrase(a, tokens(k)) for k in keys):
        return True
    return SequenceMatcher(None, norm(answer), norm(en)).ratio() >= 0.85


def score(answer: str, en: str, keys: list[str] | None) -> str:
    """'ok' | 'near' | 'no'. near = 핵심 내용어(4글자 이상)가 다 비슷하게 있음(받아쓰기 오차: denoise→denounce).
    내용어가 둘 이하면 하나도 빠지면 안 되고, 셋 이상인 문장째 핵심이면 하나는 빠져도 된다('really' 빠짐).
    11-22 첫 실전: 10개 중 2개만 정답 처리 — 상당수가 받아쓰기 오류였다."""
    if grade(answer, en, keys):
        return "ok"
    a = tokens(answer)
    if not a or not keys:
        return "no"
    for k in keys:
        kt = tokens(k)
        content = [t for t in kt if len(t) >= 4] or kt
        miss = sum(1 for t in content if max((SequenceMatcher(None, t, x).ratio() for x in a), default=0) < 0.6)
        if miss > (1 if len(content) >= 3 else 0):  # 내용어가 셋 이상인 긴 핵심만 하나 봐준다('email … members'는 안 됨)
            return "no"
    return "near"


def used(text: str, keys: list[str]) -> bool:
    """대화 속 한 문장에 그 표현을 썼나(핵심이 모두 있어야)."""
    t = tokens(text)
    return bool(keys) and all(has_phrase(t, tokens(k)) for k in keys)


def diff_keys(said: str, better: str) -> list[str]:
    """내가 틀리게 말한 문장과 고친 문장을 비교해 고친 부분(핵심)을 뽑는다. 붙어 있는 구간은 합친다."""
    s, b = tokens(said), tokens(better)
    spans = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, s, b).get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            if spans and j1 - spans[-1][1] <= 1:
                spans[-1][1] = j2
            else:
                spans.append([j1, j2])
        elif tag == "delete" and not spans and 0 < j1 < len(b):  # 빼기만 한 교정(go to abroad → go abroad): 뺀 자리 양옆
            spans.append([j1 - 1, j1 + 1])
    for sp in spans:  # 한 글자짜리 핵심('it')은 너무 약하다(아무 대답에나 들어 있음) → 옆 낱말까지
        if sp[1] - sp[0] == 1 and len(b[sp[0]]) <= 3:
            if sp[1] < len(b):
                sp[1] += 1
            elif sp[0] > 0:
                sp[0] -= 1
    out = [" ".join(b[a:c]) for a, c in spans]
    out = [x for x in out if x]
    return out[:2] or ([" ".join(b)] if b else [])


def cloze(en: str, keys: list[str]) -> str:
    """핵심 낱말 자리를 ___ 로 가린 문장(소리로 읽을 땐 'blank')."""
    words = en.split()
    norm_words = [norm(w) for w in words]
    hide = set()
    for k in keys or []:
        kt = tokens(k)
        for i in range(len(words)):
            seq, j, idx = [], i, 0
            while j < len(words) and idx < len(kt):
                parts = norm_words[j].split()
                if not parts:
                    j += 1
                    continue
                if parts == kt[idx:idx + len(parts)]:
                    seq.append(j)
                    idx += len(parts)
                    j += 1
                else:
                    break
            if idx == len(kt) and seq:
                hide.update(seq)
                break
    if not hide:
        return en
    out = []
    for i, w in enumerate(words):
        if i in hide:
            tail = re.sub(r"^[\w'’-]+", "", w)
            out.append("___" + tail)
        else:
            out.append(w)
    return " ".join(out)
