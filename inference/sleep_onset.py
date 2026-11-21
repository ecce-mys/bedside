"""Sleep onset from a laptop's senses: light, input, lock state and room loudness. No camera, no recording, no wearable.

Input: one sample every ~30 s through the night (see sensors/room_sensor.swift), e.g.
    {"t": 1795280000, "lux": 0.0, "idle": 812.0, "locked": true, "display_asleep": true, "spikes": 0}
where "spikes" is how many times in that 30 s the room got clearly louder than its own background (from the mic's
RMS level only; audio is never stored).

Rules (each one is there because a real night broke the previous version):
  rest      = dark AND the Mac was put away (locked or display off). When the display is off the light sensor stops
              reporting, so "no lux + locked/display off" counts as dark.
  runs      = rest samples glued together across gaps up to 15 min. A recording gap over 3 min means the Mac itself was
              asleep (lid closed, carried in a bag): never glue across it. (A laptop in a backpack is dark and locked,
              and once looked exactly like sleep at 20:50.)
  sleep-like = the run started at night (not 09:00-19:00) and the person was using the Mac within the 30 minutes before
              it started. (A Mac left locked at home all day once produced a "sleep onset" at 15:13.)
  lights out = the later of: the last bright sample, and the last input before the run.
  onset     = the first time, at least `latency_min` after lights out, that begins a quiet 15-minute window
              (sum of spikes <= 3 over >= 20 samples). Without mic data: lights out + latency.
  wake      = 3+ hours after onset, the first non-rest stretch longer than 5 minutes.

Measured against what I reported the next morning: +6 min and +5 min on two nights, "about right" on two more.
"""
from __future__ import annotations

from datetime import datetime

DEFAULTS = {"dark_lux": 50.0, "latency_min": 10, "away_s": 300, "explicit_off": True}
# dark_lux: measured in one bedroom: lights on ~390 lux, lights off with the screen on 10-35 lux.


def rest(r: dict, s: dict) -> bool:
    if r.get("_busy"):  # e.g. a bedtime voice chat: awake by definition
        return False
    dark = r["lux"] <= s["dark_lux"] if r.get("lux") is not None else bool(r.get("display_asleep") or r.get("locked"))
    off = r.get("locked") or r.get("display_asleep")
    away = off if s.get("explicit_off", True) else (off or r.get("idle", 0) >= s["away_s"])
    return bool(dark and away)


def in_use(r: dict, s: dict) -> bool:
    return not r.get("locked") and not r.get("display_asleep") and r.get("idle", 0) < s["away_s"]


def sleep_like(samples: list[dict], run: tuple, s: dict) -> bool:
    first = run[0]
    if 9 <= datetime.fromtimestamp(first).hour < 19:  # naps and empty rooms
        return False
    return any(first - 30 * 60 <= r["t"] <= first and in_use(r, s) for r in samples)


def runs(samples: list[dict], s: dict, gap_s: int = 15 * 60) -> list[tuple[int, int, int]]:
    """[(start t, end t, number of breaks)]."""
    out: list[list[int]] = []
    prev, cut = None, False
    for r in samples:
        if prev is not None and r["t"] - prev > 180:  # the Mac itself was asleep: lid closed
            cut = True
        prev = r["t"]
        if not rest(r, s):
            continue
        if out and not cut and r["t"] - out[-1][1] <= gap_s:
            if r["t"] - out[-1][1] > 90:
                out[-1][2] += 1
            out[-1][1] = r["t"]
        else:
            out.append([r["t"], r["t"], 0])
            cut = False
    return [tuple(x) for x in out]


def analyze(samples: list[dict], s: dict | None = None, busy: list[tuple[int, int]] | None = None) -> dict:
    s = {**DEFAULTS, **(s or {})}
    samples = sorted(samples, key=lambda r: r["t"])
    if busy:
        samples = [{**r, "_busy": True} if any(a <= r["t"] <= b for a, b in busy) else r for r in samples]
    res: dict = {"samples": len(samples)}
    if not samples:
        return res
    long_runs = [x for x in runs(samples, s) if x[1] - x[0] >= 20 * 60 and sleep_like(samples, x, s)]
    if not long_runs:
        return res
    first, end, breaks = max(long_runs, key=lambda x: x[1] - x[0])  # the night's main sleep = longest rest
    r0 = next(r for r in samples if r["t"] == first)
    bright = [r["t"] for r in samples if r["t"] < first and (r.get("lux") or 0) > s["dark_lux"]]
    lights = min(first, max(first - r0.get("idle", 0), (bright[-1] + 30) if bright else 0))
    lights = max([lights] + [b for _, b in busy or [] if b <= first])
    res["lights_out"], res["breaks"] = lights, breaks
    start = lights + s["latency_min"] * 60
    onset = start
    with_mic = [r for r in samples if lights <= r["t"] <= end and "spikes" in r]
    if with_mic:
        onset = None
        for r in with_mic:
            if r["t"] < start:
                continue
            win = [x.get("spikes", 0) for x in with_mic if r["t"] <= x["t"] < r["t"] + 15 * 60]
            if len(win) >= 20 and sum(win) <= 3:
                onset = r["t"]
                break
        if onset is None:
            onset, res["onset_note"] = start, "no quiet 15 minutes found: lights out + latency"
    res["onset"] = onset
    wake, streak = None, None
    for r in samples:
        if r["t"] < onset + 3 * 3600:
            continue
        if rest(r, s):
            streak = None
            continue
        streak = streak or r["t"]
        if r["t"] - streak >= 5 * 60:
            wake = streak
            break
    res["wake"] = wake or (end if end < samples[-1]["t"] else None)
    if res["wake"]:
        res["slept_h"] = round((res["wake"] - onset) / 3600, 2)
    return res
