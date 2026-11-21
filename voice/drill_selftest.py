# /// script
# requires-python = ">=3.12"
# dependencies = ["websockets>=13"]
# ///
"""드릴 모드 소리 없는 시험(11-22): 대화 앱 대신 음성 서버에 접속해, 합성 음성(say)으로 대답한다.

앱처럼 행동한다: setup → kickoff, 받은 소리를 재생하는 동안(+0.4초)은 마이크를 막고, 그 뒤 0.8초 생각하고 말한다.
--wrong <id> 표현의 시험은 일부러 틀리게 답한다(재시도에선 맞게). 끝나면 오간 말을 출력한다.

  uv run --script drill_selftest.py ws://127.0.0.1:PORT plan.json [--wrong ID]
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

SILENCE = b"\x00\x00" * 1600  # 0.1초(16kHz)


def tts16(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        subprocess.run(["/usr/bin/say", "-v", "Samantha", "-o", str(p), "--file-format=WAVE",
                        "--data-format=LEI16@16000", text], check=True)
        with wave.open(str(p)) as w:
            return w.readframes(w.getnframes())


def decide(coach: str, steps: list[dict], wrong: str, answered: set) -> str | None:
    c = coach.strip()
    if not c or c.startswith(("Yes.", "Right.", "Good.", "Nice", "Done.", "Okay.")):
        return None
    if "say ready" in c.lower():
        return "Ready."
    by_en = [s for s in steps if s.get("en")]
    if c.endswith(("Once more.", "Say it.")):
        s = next((s for s in by_en if s["en"] in c), None)
        return s["en"] if s else None
    if c.startswith("Fix this."):
        s = next((s for s in steps if s.get("said") and s["said"] in c), None)
        return s["en"] if s else None
    if c.startswith("Last one"):
        s = next(s for s in steps if s["op"] == "learn")
        return "Today " + s["en"][0].lower() + s["en"][1:]
    s = next((s for s in steps if s.get("ko") and s["ko"] in c and s["op"] == "test"), None)
    if s:
        if s["id"] == wrong and s["id"] not in answered:
            answered.add(s["id"])
            return "Something completely different, sorry."
        return s["en"]
    return None


async def main(url: str, plan_path: str, wrong: str) -> int:
    import websockets
    steps = json.loads(Path(plan_path).read_text())["steps"]
    lines: list[str] = []
    answered: set = set()
    q: asyncio.Queue = asyncio.Queue()
    mic_open_at = [0.0]
    playhead = [0.0]
    got_end = False
    async with websockets.connect(url, max_size=16 << 20) as ws:
        await ws.send(json.dumps({"setup": {
            "systemInstruction": {"parts": [{"text": "<<DRILL>>" + plan_path}]},
            "realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": 1300, "prefixPaddingMs": 300}}}}))

        async def mic():
            while True:
                if time.time() < mic_open_at[0]:  # 재생 중엔 마이크 막힘(앱처럼)
                    await asyncio.sleep(0.05)
                    continue
                try:
                    pcm = q.get_nowait()
                except asyncio.QueueEmpty:
                    pcm = SILENCE
                for i in range(0, len(pcm), 3200):
                    await ws.send(json.dumps({"realtimeInput": {"audio": {
                        "mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(pcm[i:i + 3200]).decode()}}}))
                    await asyncio.sleep(0.1)

        mic_task = None
        coach, audio_s = "", 0.0
        async for raw in ws:
            o = json.loads(raw)
            if "setupComplete" in o:
                await ws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "(start)"}]}],
                                                            "turnComplete": True}}))
                mic_task = asyncio.create_task(mic())
                continue
            if "toolCall" in o:
                await ws.send(json.dumps({"toolResponse": {"functionResponses": [{"id": "end", "name": "end_session",
                                                                                   "response": {"ok": True}}]}}))
                got_end = True
                lines.append("[end_session]")
                break
            sc = o.get("serverContent", {})
            if "outputTranscription" in sc:
                coach += sc["outputTranscription"]["text"]
            if "inputTranscription" in sc:
                lines.append(f"    (서버가 들음: {sc['inputTranscription']['text']})")
            for part in (sc.get("modelTurn") or {}).get("parts", []):
                audio_s += len(base64.b64decode(part["inlineData"]["data"])) / 2 / 24000
            if sc.get("turnComplete"):
                start = max(time.time(), playhead[0])
                playhead[0] = start + audio_s
                mic_open_at[0] = playhead[0] + 0.4
                if coach.strip():
                    lines.append(f"AI: {coach.strip()}  [{audio_s:.1f}s]")
                ans = decide(coach, steps, wrong, answered)
                coach, audio_s = "", 0.0
                if ans:
                    lines.append(f"나: {ans}")
                    pcm = await asyncio.to_thread(tts16, ans)
                    await q.put(b"\x00\x00" * 12800 + pcm)  # 0.8초 생각하고 말하기
        if mic_task:
            mic_task.cancel()
    print("\n".join(lines))
    return 0 if got_end else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("plan")
    ap.add_argument("--wrong", default="")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(asyncio.wait_for(main(a.url, a.plan, a.wrong), 400)))
