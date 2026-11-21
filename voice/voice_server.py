# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx-whisper",
#   "mlx-audio",
#   "misaki[en]",
#   "numpy",
#   "websockets>=13",
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Claude 음성 서버(11-19 사용자: "헤이 클로드"로 Claude 음성 대화, Gemini 연결 대신).

Claude API에는 음성 입출력이 없다 → 내 말 끝 감지(소리 크기) → Whisper(로컬, large-v3-turbo) → Claude(구독 CLI를 대화 내내
띄워 두고 stream-json으로 주고받음) → Kokoro(로컬 음성, 24kHz) 를 이어 붙인다.

대화 앱(SleepSense --converse, morning.swift)은 Gemini Live 방식으로 말한다. 이 서버가 같은 방식(setup·realtimeInput·
clientContent ↔ setupComplete·serverContent·toolCall)으로 받아 주므로, 앱은 접속 주소만 바꾸면 마이크·에코 처리·기록·
끝내기를 그대로 쓴다.

호출어: --wake 이면 ~/.odyssey/morning/wake_port 에 UDP 포트를 적는다 → 수면 센서 앱이 마이크를 켜 둔 동안(밤 창 + 맥을
안 쓸 때) 같은 소리를 16kHz로 보낸다 → 짧은 말소리 구간만 받아써 "hey claude / 헤이 클로드"면 `ody morning wake`를 실행한다.
소리는 어디에도 저장하지 않는다(메모리에서 받아쓰고 버림).

  uv run --script claude_live.py [--wake] [--model sonnet] [--voice af_heart]
상태: ~/.odyssey/morning/live.json {pid, ws_port, udp_port, ready}, 기록: ~/.odyssey/morning/live.log
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

DIR = Path.home() / ".odyssey" / "morning"
STATE = DIR / "live.json"
WAKE_PORT = DIR / "wake_port"
STT_MODEL = "mlx-community/whisper-large-v3-turbo"
TTS_MODEL = "prince-canuma/Kokoro-82M"
SR_IN, SR_OUT = 16000, 24000
END = "<<END>>"

VOICE_RULES = f"""

How this works (voice): everything you write is read aloud by a text-to-speech voice, and what they say reaches you as speech-to-text.
- Write only the words to be spoken, in English. No emoji, no markdown, no lists, no stage directions like (laughs).
- The transcript of what they said may have small recognition errors; guess the sentence they meant. Never correct a "mistake" that is only a transcription error (odd spelling, homophones, missing punctuation).
- Notes in parentheses from the user turn are instructions for you, not words they said.
- You can't call tools here. Where the instructions say to call end_session, instead end your final message with the exact token {END}."""

WAKE_RE = re.compile(r"\b(hey|hi|okay|ok|hay)[\s,.!-]*(claude|clod|claud|clawd|cloud|klaud|clyde|claude's|clode)\b"
                     r"|헤이\s*(클로드|클로우드|클라우드|클러드|글로드)|에이\s*클로드", re.I)
HALLUCINATIONS = {"thank you.", "thank you", "thanks for watching!", "thanks for watching.", "you", "bye.", "bye",
                  "시청해 주셔서 감사합니다.", "감사합니다.", ".", "..."}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------- 소리 ----------------
class Vad:
    """소리 크기로 말 구간을 자른다(조용한 방 기준). 바탕 소리를 따라가며 그보다 충분히 크면 말소리로 본다."""

    def __init__(self, silence_ms: int = 2000, prefix_ms: int = 300, min_speech_s: float = 0.35, max_s: float = 40):
        self.silence = silence_ms / 1000
        self.prefix = prefix_ms / 1000
        self.min_speech = min_speech_s
        self.max_s = max_s
        self.floor = -60.0
        self.buf: list[np.ndarray] = []      # 지금 말 구간
        self.pre: list[np.ndarray] = []      # 말 시작 전 여유분
        self.speaking = False
        self.voiced = 0.0
        self.quiet = 0.0
        self.run = 0
        self.t_start = 0.0
        self.starts: list[float] = []        # 끝난 말 구간마다 말을 시작한 시각(드릴 반응 시간용, 꺼내 쓰는 쪽이 비운다)

    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        """int16 → 끝난 말 구간(float32) 목록."""
        out = []
        x = pcm.astype(np.float32) / 32768
        for i in range(0, len(x), 480):  # 30ms
            f = x[i:i + 480]
            if len(f) == 0:
                continue
            db = 10 * np.log10(float(np.mean(f * f)) + 1e-12)
            loud = db > max(-48.0, self.floor + 12)
            if not loud:  # 바탕 소리는 천천히 따라간다
                self.floor = 0.98 * self.floor + 0.02 * db
            dt = len(f) / SR_IN
            if not self.speaking:
                self.pre.append(f)
                while sum(len(p) for p in self.pre) / SR_IN > self.prefix:
                    self.pre.pop(0)
                self.run = self.run + 1 if loud else 0
                if self.run >= 3:
                    self.speaking, self.buf, self.voiced, self.quiet = True, list(self.pre), 0.0, 0.0
                    self.t_start = time.time() - 0.09  # 소리 3칸(30ms씩)이 이어져야 말로 보므로 그만큼 앞
                    self.pre = []
                continue
            self.buf.append(f)
            if loud:
                self.voiced += dt
                self.quiet = 0.0
            else:
                self.quiet += dt
            total = sum(len(b) for b in self.buf) / SR_IN
            if self.quiet >= self.silence or total >= self.max_s:
                if self.voiced >= self.min_speech:
                    out.append(np.concatenate(self.buf))
                    self.starts.append(self.t_start)
                self.speaking, self.buf, self.run = False, [], 0
        return out

    def flush(self) -> list[np.ndarray]:
        """소리가 끊겼다(앱이 보내기를 멈춤·파일 끝) → 말하던 중이면 거기서 끝난 걸로 본다."""
        out = []
        if self.speaking and self.voiced >= self.min_speech:
            out.append(np.concatenate(self.buf))
            self.starts.append(self.t_start)
        self.speaking, self.buf, self.run = False, [], 0
        return out

    def reset(self):
        self.speaking, self.buf, self.pre, self.run = False, [], [], 0


class Ears:
    def __init__(self):
        import mlx_whisper
        self.w = mlx_whisper
        self.transcribe(np.zeros(SR_IN, dtype=np.float32))  # 모델 올리기

    def transcribe(self, audio: np.ndarray, language: str | None = None, strict: bool = True, prompt: str | None = None) -> str:
        """strict=False(드릴): 말할 차례에 들어온 짧은 대답('Ready.' 한 마디)이 '말 없음'으로 걸러지지 않게 느슨하게.
        prompt: 기대하는 낱말(드릴의 핵심 표현) — 'denoise'·'lab members' 같은 말을 엉뚱한 낱말로 받아쓰지 않게."""
        ns, lp = (0.6, -1.2) if strict else (0.95, -1.6)
        r = self.w.transcribe(audio, path_or_hf_repo=STT_MODEL, language=language, temperature=0.0,
                              condition_on_previous_text=False, no_speech_threshold=ns, verbose=None,
                              **({"initial_prompt": prompt} if prompt else {}))
        segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < ns and s.get("avg_logprob", 0) > lp]
        text = " ".join(s["text"].strip() for s in segs).strip()
        return "" if text.lower() in HALLUCINATIONS else text


class Mouth:
    def __init__(self, voice: str, speed: float):
        from mlx_audio.tts.utils import load_model
        self.model = load_model(TTS_MODEL)
        self.voice, self.speed = voice, speed
        self.say("Hi.")  # 모델 올리기

    def say(self, text: str) -> bytes:
        chunks = [np.array(r.audio, dtype=np.float32).reshape(-1)
                  for r in self.model.generate(text=text, voice=self.voice, speed=self.speed, lang_code="a")]
        if not chunks:
            return b""
        a = np.clip(np.concatenate(chunks), -1, 1)
        return (a * 32767).astype("<i2").tobytes()


def say_ko(text: str, voice: str = "Yuna", rate: int = 175) -> bytes:
    """한국어 음성(Kokoro엔 한국어가 없다) → macOS say, 24kHz 16bit 모노 PCM."""
    import wave
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "ko.wav"
        subprocess.run(["/usr/bin/say", "-v", voice, "-r", str(rate), "-o", str(out), "--file-format=WAVE",
                        "--data-format=LEI16@24000", text], check=True, timeout=30, capture_output=True)
        with wave.open(str(out)) as w:
            pcm = w.readframes(w.getnframes())
            if w.getnchannels() == 2:
                pcm = np.frombuffer(pcm, dtype="<i2")[::2].tobytes()
    return pcm


def clean(text: str) -> str:
    text = re.sub(r"\([^)]*\)|\*+|_+|#+|`+", " ", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    text = re.sub(r"[가-힣ㄱ-ㅎㅏ-ㅣ]+", "", text)  # 영어 음성이라 한글은 못 읽는다
    return re.sub(r"\s+", " ", text).strip()


# ---------------- 머리(Claude) ----------------
class Brain:
    """claude 구독 CLI를 대화 내내 띄워 두고 한 턴씩 보낸다. 실패하면 로컬 LLM(OpenAI 방식)으로."""

    def __init__(self, system: str, model: str):
        self.system = system + VOICE_RULES
        self.model = model
        self.p: asyncio.subprocess.Process | None = None
        self.history: list[dict] = []  # 로컬 대체용
        self.cwd = tempfile.mkdtemp(prefix="claude-live-")
        self.engine = "claude"

    async def start(self):
        exe = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
        self.p = await asyncio.create_subprocess_exec(
            exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--include-partial-messages",
            "--verbose", "--model", self.model, "--system-prompt", self.system, "--tools", "", "--strict-mcp-config",
            "--setting-sources", "", "--no-session-persistence", "--disable-slash-commands",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.cwd, limit=16 << 20)

    async def reply(self, text: str):
        """글자 조각을 내보낸다."""
        self.history.append({"role": "user", "content": text})
        got = ""
        if self.engine == "claude" and self.p and self.p.returncode is None:
            try:
                self.p.stdin.write((json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n").encode())
                await self.p.stdin.drain()
                while True:
                    line = await asyncio.wait_for(self.p.stdout.readline(), 40)
                    if not line:
                        break
                    o = json.loads(line)
                    if o.get("type") == "stream_event":
                        ev = o.get("event", {})
                        if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                            got += ev["delta"]["text"]
                            yield ev["delta"]["text"]
                    elif o.get("type") == "result":
                        if o.get("is_error") and not got:
                            log("claude 오류:", str(o.get("result"))[:200])
                            break
                        self.history.append({"role": "assistant", "content": got})
                        return
            except (asyncio.TimeoutError, OSError, ValueError) as e:
                log("claude 실패:", repr(e)[:200])
            if got:
                self.history.append({"role": "assistant", "content": got})
                return
            self.engine = "local"
        async for piece in self._local():
            got += piece
            yield piece
        self.history.append({"role": "assistant", "content": got})

    async def _local(self):
        """구독이 끊겼거나 실패: 로컬 OpenAI 방식 서버(Strata 8080 → oMLX 8001)."""
        msgs = [{"role": "system", "content": self.system}] + self.history[-30:]
        for port in (8080, 8001):
            try:
                def call():
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                                 data=json.dumps({"messages": msgs, "max_tokens": 300}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        return json.loads(r.read())["choices"][0]["message"]["content"]
                text = await asyncio.to_thread(call)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
                log(f"로컬 LLM({port})으로 대답")
                yield text
                return
            except Exception:
                continue
        yield f"Sorry, I lost my connection for a moment. Let's pick this up later. Good night. {END}"

    async def close(self):
        if self.p and self.p.returncode is None:
            try:
                self.p.stdin.close()
                await asyncio.wait_for(self.p.wait(), 5)
            except Exception:
                self.p.kill()
        shutil.rmtree(self.cwd, ignore_errors=True)


# ---------------- 드릴(11-22, Morning Words) ----------------
DRILL_MARK = "<<DRILL>>"


class Drill:
    """setup 지시문이 '<<DRILL>>계획경로'면 대화 대신 드릴을 돈다(odyssey/words.py가 계획을 쓴다).
    묻기(한국어 뜻 = macOS 유나, 영어 = Kokoro) → 듣기(받아쓰기) → 채점(wordgrade) → 반응 시간 → 결과 파일에 한 줄씩.
    LLM을 안 쓴다: 구독이 끊겨도, 망이 없어도 돈다."""
    GAP = b"\x00\x00" * int(SR_OUT * 0.3)

    def __init__(self, server: "Server", plan_path: str, send, speak_lock: asyncio.Lock, speaking):
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 wordgrade.py(채점)
        import wordgrade
        self.g = wordgrade
        self.s, self.send, self.lock, self.speaking = server, send, speak_lock, speaking
        self.p = json.loads(Path(plan_path).read_text())
        self.res = Path(self.p["results"])
        self.q: asyncio.Queue = asyncio.Queue()   # (말 구간, 말 시작 시각)
        self.stop: str | None = None
        self.cache: dict = {}
        self.tts_lock = asyncio.Lock()
        self.play_until = 0.0                      # 앱이 받은 소리를 다 재생할 예상 시각(연달아 보내면 줄을 선다)
        self.misses = 0

    def record(self, **row):
        row["t"] = time.time()
        self.res.parent.mkdir(parents=True, exist_ok=True)
        with open(self.res, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def pcm(self, lang: str, text: str) -> bytes:
        k = (lang, text)
        if k not in self.cache:
            async with self.tts_lock:
                if k not in self.cache:
                    if lang == "ko":
                        self.cache[k] = await asyncio.to_thread(say_ko, text, self.p.get("ko_voice", "Yuna"),
                                                                int(self.p.get("ko_rate", 175)))
                    else:
                        self.cache[k] = await asyncio.to_thread(self.s.mouth.say, clean(text)) if clean(text) else b""
        return self.cache[k]

    def cues(self, st: dict) -> list[tuple[str, str]]:
        op = st["op"]
        if op == "learn":
            if st.get("ko"):
                return [("ko", st["ko"]), ("en", st["en"]), ("en", "Say it.")]
            return [("en", f"You said: {st.get('said', '')}."), ("en", f"Better: {st['en']}"), ("en", "Say it.")]
        if op == "test":
            out = [("ko", st["ko"])]
            if "___" in (st.get("cloze") or ""):  # 11-22: 'blank'만 읽으니 그대로 따라 말함 → 채우라고 먼저 말한다
                out.append(("en", "Fill in. " + st["cloze"].replace("___", "blank")))
            return out
        if op == "fix":
            return [("en", "Fix this."), ("en", st.get("said") or "")]
        if op == "free":  # 11-22: 'Tell me one thing you'll do today…'는 못 알아들었다("What is this?")
            return [("en", "Last one. Make your own sentence with one of today's phrases.")]
        return []

    @staticmethod
    def hint(st: dict) -> str:
        """받아쓰기 힌트: 그 문제의 핵심 낱말(배우기·다시 말하기는 문장째)."""
        if st.get("op") == "learn":
            return st.get("en", "")
        return ", ".join(st.get("key") or st.get("keys") or [])

    async def prerender(self):
        """'ready'를 기다리는 동안 소리를 미리 만들어 둔다(묻는 사이 공백이 없게)."""
        for st in self.p["steps"]:
            pieces = self.cues(st) + ([("en", st["en"]), ("en", "Yes. " + st["en"]), ("en", "Right. " + st["en"])]
                                      if st.get("en") else [])
            for lang, text in pieces:
                if self.stop:
                    return
                try:
                    await self.pcm(lang, text)
                except Exception as e:
                    log("드릴 소리 만들기 실패:", repr(e)[:120])

    async def play(self, pieces: list[tuple[str, str]]) -> float:
        """조각들을 보내고 앱이 재생을 끝낼 예상 시각을 돌려준다."""
        async with self.lock:
            total = 0.0
            for n, (lang, text) in enumerate(pieces):
                pcm = await self.pcm(lang, text)
                if not pcm:
                    continue
                if n:
                    pcm = self.GAP + pcm
                await self.send({"serverContent": {"outputTranscription": {"text": text + " "}}})
                for i in range(0, len(pcm), SR_OUT):  # 0.5초씩
                    await self.send({"serverContent": {"modelTurn": {"parts": [{"inlineData": {
                        "mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(pcm[i:i + SR_OUT]).decode()}}]}}})
                total += len(pcm) / 2 / SR_OUT
            await self.send({"serverContent": {"turnComplete": True}})
            start = max(time.time(), self.play_until)
            self.play_until = start + total
            return self.play_until

    async def hear(self, end_at: float, wait_s: float, lang: str | None = "en", prompt: str | None = None,
                   alt: str | None = None) -> tuple[str, float | None]:
        """재생이 끝난 뒤 wait_s 안에 시작한 말을 받아쓴다(말하는 중이면 15초까지 더 기다림). (글, 반응 초)
        alt: 같은 소리를 다른 언어로도 받아써 뒤에 붙인다(시작 대기: '레디'·'응')."""
        deadline = end_at + wait_s
        while True:
            now = time.time()
            if (now >= deadline and not self.speaking()) or now >= deadline + 15 or self.stop:
                return "", None
            try:
                seg, ts = await asyncio.wait_for(self.q.get(), 0.3)
            except asyncio.TimeoutError:
                continue
            if ts < end_at - 0.3:  # 코치 소리가 끝나기 전에 시작된 소리(메아리·잡음)
                log(f"드릴: 재생 중 시작된 소리 버림({len(seg) / SR_IN:.1f}s)")
                continue
            async with self.tts_lock:  # MLX(받아쓰기·Kokoro)를 두 스레드에서 동시에 돌리지 않는다
                text = await asyncio.to_thread(self.s.ears.transcribe, seg, lang, False, prompt)
                if alt:
                    other = await asyncio.to_thread(self.s.ears.transcribe, seg, alt, False)
                    text = f"{text} {other}".strip()
            log(f"드릴 들음({len(seg) / SR_IN:.1f}s): {text[:80]!r}")
            if not text:
                continue
            await self.send({"serverContent": {"inputTranscription": {"text": text}}})
            return text, round(max(0.0, ts - end_at), 2)

    def stopped(self, heard: str) -> bool:
        if heard and self.g.STOP_RE.search(heard):
            self.stop = "user"
        return bool(self.stop)

    async def ready(self, end_at: float) -> bool:
        """11-22 첫 실전: 언어 자동 감지가 'ready'를 카탈루냐어·스페인어로 받아써 80초 허비 → 영어로 받아쓰고(힌트 Ready)
        한국어로도 한 번 더('레디'·'응'·'시작'). 다시 말해 달라는 말은 45초에 한 번만."""
        deadline = end_at + float(self.p.get("ready_s", 900))
        last_nudge = time.time()
        while time.time() < deadline and not self.stop:
            heard, _ = await self.hear(end_at, max(1.0, deadline - end_at), lang="en",
                                       prompt="Ready. Okay. Yes. Let's go.", alt="ko")
            if not heard:
                return False
            if self.g.READY_RE.search(heard):
                self.record(op="ready", heard=heard)
                return True
            if self.stopped(heard):
                return False
            if time.time() - last_nudge > 45:
                last_nudge = time.time()
                end_at = await self.play([("en", "Just say ready when you want to start.")])
            else:
                end_at = time.time()
        return False

    async def run(self):
        n = ok_n = 0
        lats: list[float] = []
        pre = asyncio.create_task(self.prerender())
        try:
            end = await self.play([("en", self.p.get("intro") or "Morning. Say ready when you are.")])
            self.record(op="intro")
            if not await self.ready(end):
                self.stop = self.stop or "no_ready"
            slow = float(self.p.get("slow_s", 4.0))
            for st in self.p["steps"] if self.stop is None else []:
                op = st["op"]
                end = await self.play(self.cues(st))
                hint = self.hint(st)
                if op == "learn":
                    heard, lat = await self.hear(end, float(self.p.get("learn_listen_s", 10)), prompt=hint)
                    if self.stopped(heard):
                        break
                    ok = bool(heard) and self.g.score(heard, st["en"], st.get("key")) != "no"
                    self.record(op="learn", id=st["id"], ok=ok, lat=lat, heard=heard)
                    await self.play([("en", "Good.")] if ok else [("en", st["en"])])
                elif op in ("test", "fix"):
                    heard, lat = await self.hear(end, float(self.p.get("listen_s", 9)), prompt=hint)
                    if self.stopped(heard):
                        break
                    passed = bool(heard) and bool(self.g.PASS_RE.search(heard))
                    sc = "no" if not heard or passed else self.g.score(heard, st["en"], st.get("key"))
                    ok = sc != "no"
                    self.record(op=op, id=st["id"], ok=ok, near=sc == "near", lat=lat if ok else None, heard=heard,
                                lvl=st.get("lvl"), timeout=not heard)
                    n += 1
                    if ok:
                        ok_n += 1
                        lats.append(lat or 0.0)
                        lead = "Close. " if sc == "near" else "Yes. " if (lat or 0) <= slow else "Right. "
                        await self.play([("en", lead + st["en"])])
                    else:  # 정답을 들려주고 한 번 더 말하게(기록은 처음 대답으로)
                        e2 = await self.play([("en", st["en"]), ("en", "Once more.")])
                        h2, _ = await self.hear(e2, float(self.p.get("listen_s", 9)), prompt=st.get("en"))
                        if self.stopped(h2):
                            break
                        if h2:
                            ok2 = self.g.score(h2, st["en"], st.get("key")) != "no"
                            self.record(op="retry", id=st["id"], ok=ok2, heard=h2)
                            if ok2:
                                await self.play([("en", "Good.")])
                elif op == "free":
                    heard, lat = await self.hear(end, float(self.p.get("free_listen_s", 25)), prompt=hint)
                    if self.stopped(heard):
                        break
                    keys = [k for k in st.get("keys", []) if heard and self.g.used(heard, [k])]
                    self.record(op="free", ok=bool(keys), heard=heard, used_keys=keys)
                    if heard:
                        await self.play([("en", f"Nice, you used {keys[0]}." if keys else "Good. Try to use one of them tonight.")])
                else:
                    continue
                self.misses = 0 if heard else self.misses + 1
                if self.misses >= 4:  # 네 번 연달아 대답 없음 = 자리 비움·잠듦
                    self.stop = "no_answer"
                    break
            if n:
                avg = sum(lats) / len(lats) if lats else None
                end_line = [("en", f"Done. {ok_n} out of {n}" + (f", about {avg:.1f} seconds each" if avg else "") + ".")]
                if self.p.get("question"):  # 덩어리 15분(week.py): 등굣길에 붙잡고 걸을 질문 하나
                    end_line.append(("en", "Something to chew on today. " + self.p["question"]))
                await self.play(end_line + [("en", "See you tonight.")])
            elif self.stop == "user":
                await self.play([("en", "Okay. Later, then.")])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("드릴 오류:", repr(e)[:200])
            self.stop = f"error: {e!r}"[:200]
        finally:
            pre.cancel()
        self.record(op="end", why=self.stop or "done", n=n, ok=ok_n)
        await self.send({"toolCall": {"functionCalls": [{"id": "end", "name": "end_session", "args": {}}]}})
        await self.send({"serverContent": {"turnComplete": True}})


# ---------------- 서버 ----------------
class Server:
    def __init__(self, args):
        self.args = args
        self.ears: Ears | None = None
        self.mouth: Mouth | None = None
        self.busy = False            # 대화 중
        self.last_session_end = 0.0
        self.wake_cool = 0.0
        self.ws_port = 0
        self.udp_port = 0
        self.claude_ok = False

    # --- 대화 한 번 ---
    async def session(self, ws):
        if self.busy:
            await ws.close(1013, "busy")
            return
        self.busy = True
        log("대화 연결")
        brain: Brain | None = None
        drill: Drill | None = None
        drill_task: asyncio.Task | None = None
        speak_lock = asyncio.Lock()
        utter: asyncio.Queue = asyncio.Queue()
        vad = Vad()
        ended = False

        def route(segs):  # 끝난 말 구간 → 대화(받아쓰기→대답) 또는 드릴(시작 시각과 함께)
            for seg in segs:
                ts = vad.starts.pop(0) if vad.starts else time.time()
                if drill:
                    drill.q.put_nowait((seg, ts))
                else:
                    utter.put_nowait(seg)

        async def send(o):
            await ws.send(json.dumps(o))

        async def speak_turn(text: str):
            nonlocal ended
            async with speak_lock:
                if ended:
                    return
                buf, end_now = "", False
                t0, first = time.time(), [None, None]
                async def flush(sentence: str):
                    s = clean(sentence)
                    if not s:
                        return
                    if first[0] is None:
                        first[0] = time.time() - t0
                    pcm = await asyncio.to_thread(self.mouth.say, s)
                    if first[1] is None:
                        first[1] = time.time() - t0
                        log(f"첫 문장 {first[0]:.2f}s → 첫 소리 {first[1]:.2f}s ({brain.engine})")
                    await send({"serverContent": {"outputTranscription": {"text": s + " "}}})
                    step = SR_OUT * 2 // 2  # 0.5초씩
                    for i in range(0, len(pcm), step):
                        await send({"serverContent": {"modelTurn": {"parts": [{"inlineData": {
                            "mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(pcm[i:i + step]).decode()}}]}}})
                async for piece in brain.reply(text):
                    buf += piece
                    if END in buf:
                        buf, end_now = buf.replace(END, ""), True
                    # 문장이 끝나면 바로 읽는다(첫 소리를 빨리)
                    while True:
                        m = re.search(r"[.!?…](?=\s)|\n", buf)
                        if not m:
                            break
                        await flush(buf[:m.end()])
                        buf = buf[m.end():]
                if END in buf:
                    buf, end_now = buf.replace(END, ""), True
                await flush(buf)
                if end_now:
                    ended = True
                    await send({"toolCall": {"functionCalls": [{"id": "end", "name": "end_session", "args": {}}]}})
                await send({"serverContent": {"turnComplete": True}})

        async def listen():  # 끝난 말 구간 → 받아쓰기 → 대답
            while True:
                audio = await utter.get()
                t0 = time.time()
                text = await asyncio.to_thread(self.ears.transcribe, audio)
                if not text:
                    continue
                log(f"들음({time.time() - t0:.1f}s): {text[:80]}")
                await send({"serverContent": {"inputTranscription": {"text": text}}})
                await speak_turn(text)

        worker = None
        last_audio = [time.time()]

        async def stall():  # 소리가 1초 넘게 안 오면 말 끝
            while True:
                await asyncio.sleep(0.3)
                if vad.speaking and time.time() - last_audio[0] > 1.0:
                    route(vad.flush())
        staller = asyncio.create_task(stall())
        try:
            async for raw in ws:
                o = json.loads(raw)
                if "setup" in o:
                    st = o["setup"]
                    system = "".join(p.get("text", "") for p in st.get("systemInstruction", {}).get("parts", []))
                    aad = st.get("realtimeInputConfig", {}).get("automaticActivityDetection", {})
                    vad = Vad(silence_ms=int(aad.get("silenceDurationMs", 2000)), prefix_ms=int(aad.get("prefixPaddingMs", 300)))
                    if system.startswith(DRILL_MARK):  # 드릴: LLM 없이 계획대로
                        # 대답이 한두 낱말('Ready.', 'let down')이라 짧은 말도 말로 친다(대화 기본 0.35초면 버려짐)
                        vad = Vad(silence_ms=int(aad.get("silenceDurationMs", 1300)), prefix_ms=int(aad.get("prefixPaddingMs", 300)),
                                  min_speech_s=0.12)
                        drill = Drill(self, system[len(DRILL_MARK):].strip(), send, speak_lock, lambda: vad.speaking)
                        log("드릴 모드")
                    else:
                        brain = Brain(system, self.args.model)
                        await brain.start()
                        worker = asyncio.create_task(listen())
                    await send({"setupComplete": {}})
                elif "realtimeInput" in o:
                    d = o["realtimeInput"].get("audio", {}).get("data")
                    last_audio[0] = time.time()
                    if d and not speak_lock.locked():  # 대답하는 동안 들어온 소리는 버린다(앱도 재생 중엔 막는다)
                        route(vad.feed(np.frombuffer(base64.b64decode(d), dtype="<i2")))
                    elif speak_lock.locked():
                        vad.reset()
                elif "clientContent" in o:
                    turns = o["clientContent"].get("turns", [])
                    text = " ".join(p.get("text", "") for t in turns for p in t.get("parts", []))
                    if drill:  # 첫 지시(kickoff) = 시작, 앱의 '조용함·너무 길어짐' 알림 = 마무리
                        if drill_task is None:
                            drill_task = asyncio.create_task(drill.run())
                        elif "gone quiet" in text or "long time" in text:
                            drill.stop = drill.stop or ("idle" if "quiet" in text else "max")
                    elif text.strip() and brain:
                        asyncio.create_task(speak_turn(text))
                # toolResponse 등은 무시
        except Exception as e:
            if "ConnectionClosed" not in type(e).__name__:  # 앱이 끝내며 그냥 닫는 건 정상
                log("대화 오류:", repr(e)[:200])
        finally:
            staller.cancel()
            if drill_task:
                drill_task.cancel()
            if worker:
                worker.cancel()
            if brain:
                if brain.engine != "claude" and self.claude_ok:  # 대화 중 Claude가 끊김 → 다음 대화는 Gemini로
                    self.claude_ok = False
                    self.write_state()
                await brain.close()
            self.busy = False
            self.last_session_end = time.time()
            log("대화 끝")

    # --- 호출어 ---
    async def wake_loop(self):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)

        class P(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                if addr[0] == "127.0.0.1" and not q.full():
                    q.put_nowait(data)

        tr, _ = await loop.create_datagram_endpoint(P, local_addr=("127.0.0.1", 0))
        self.udp_port = tr.get_extra_info("sockname")[1]
        # 센서에 포트를 알리는 건(wake_port) 데몬 몫이다: 집·시간대·대화 중 아님을 보고 쓰고 지운다(morning.live_tick)
        vad =Vad(silence_ms=600, prefix_ms=250, min_speech_s=0.3, max_s=4)
        log(f"호출어 듣는 중(UDP {self.udp_port})")
        while True:
            data = await q.get()
            if self.busy or time.time() < self.wake_cool or time.time() - self.last_session_end < 20:
                vad.reset()
                continue
            segs = vad.feed(np.frombuffer(data, dtype="<i2"))
            vad.starts.clear()  # 호출어는 시작 시각을 안 쓴다(쌓이지 않게)
            for seg in segs:
                if len(seg) / SR_IN > 4:
                    continue
                text = await asyncio.to_thread(self.ears.transcribe, seg)
                if text and WAKE_RE.search(text):
                    log(f"호출어: {text}")
                    self.wake_cool = time.time() + 60
                    subprocess.Popen(self.args.on_wake, shell=True, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                # 호출어가 아닌 말은 글자로도 남기지 않는다(방 소리 사생활)

    def write_state(self):
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "ws_port": self.ws_port, "udp_port": self.udp_port,
                                   "wake": bool(self.args.wake), "model": self.args.model, "ready": True,
                                   "claude_ok": self.claude_ok, "started": int(time.time())}))
        tmp.replace(STATE)

    async def check_claude(self) -> bool:
        """구독(로그인)이 살아 있나: 아주 짧은 한 마디를 받아 본다."""
        b = Brain("Reply with one word.", self.args.model)
        try:
            await b.start()
            got = ""
            async for piece in b.reply("Say ok."):
                got += piece
            return b.engine == "claude" and bool(got.strip())
        except Exception:
            return False
        finally:
            await b.close()

    async def main(self):
        import websockets
        log("모델 올리는 중…")
        t0 = time.time()
        self.ears = await asyncio.to_thread(Ears)
        self.mouth = await asyncio.to_thread(Mouth, self.args.voice, self.args.speed)
        self.claude_ok = await self.check_claude()
        log(f"준비 {time.time() - t0:.1f}s · Claude {'됨' if self.claude_ok else '안 됨(구독·로그인 확인) → 대화는 Gemini로'}")
        async with websockets.serve(self.session, "127.0.0.1", self.args.port, max_size=16 << 20) as srv:
            self.ws_port = srv.sockets[0].getsockname()[1]
            tasks = []
            if self.args.wake:
                tasks.append(asyncio.create_task(self.wake_loop()))
                await asyncio.sleep(0.2)
            DIR.mkdir(parents=True, exist_ok=True)
            tmp = STATE.with_suffix(".tmp")
            self.write_state()
            log(f"대기 ws://127.0.0.1:{self.ws_port}")
            await asyncio.Future()


def cleanup(*_):
    for p in (STATE, WAKE_PORT):
        try:
            if p == STATE and json.loads(p.read_text()).get("pid") != os.getpid():
                continue
            p.unlink()
        except (OSError, ValueError):
            pass
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=0.95)
    ap.add_argument("--wake", action="store_true")
    ap.add_argument("--on-wake", default="")
    ap.add_argument("--state-dir", default=str(DIR))
    a = ap.parse_args()
    DIR = Path(a.state_dir)
    STATE, WAKE_PORT = DIR / "live.json", DIR / "wake_port"
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    try:
        asyncio.run(Server(a).main())
    finally:
        cleanup()
