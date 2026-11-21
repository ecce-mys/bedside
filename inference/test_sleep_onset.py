"""Synthetic nights, including the two that broke earlier versions.  Run: python3 -m unittest test_sleep_onset"""
import unittest
from datetime import datetime

from sleep_onset import analyze


def ts(h, m, day=1):
    return int(datetime(2026, 10, 25 + day - 1, h, m).timestamp())


def night(segments, step=30):
    """segments: [((h, m, day), (h, m, day), lux, idle, locked, spikes)] -> samples every `step` seconds."""
    out = []
    for (h0, m0, d0), (h1, m1, d1), lux, idle, locked, spikes in segments:
        t = ts(h0, m0, d0)
        while t < ts(h1, m1, d1):
            r = {"t": t, "idle": idle, "locked": locked, "display_asleep": locked, "spikes": spikes}
            if not locked:  # the light sensor only reports while the display is on
                r["lux"] = lux
            out.append(r)
            t += step
    return out


class SleepOnsetTest(unittest.TestCase):
    def test_ordinary_night(self):
        s = night([((23, 0, 1), (1, 40, 2), 390, 3, False, 2),     # using the Mac, lights on
                   ((1, 40, 2), (1, 41, 2), 20, 30, False, 1),     # lights off, screen still on
                   ((1, 41, 2), (2, 5, 2), 0, 900, True, 1),       # locked, still moving around
                   ((2, 5, 2), (9, 0, 2), 0, 9000, True, 0),       # quiet
                   ((9, 0, 2), (9, 20, 2), 300, 5, False, 3)])     # up
        r = analyze(s)
        self.assertEqual(datetime.fromtimestamp(r["lights_out"]).strftime("%H:%M"), "01:40")
        self.assertAlmostEqual(r["onset"], ts(2, 5, 2), delta=5 * 60)
        self.assertEqual(datetime.fromtimestamp(r["wake"]).strftime("%H:%M"), "09:00")

    def test_laptop_in_a_bag_is_not_sleep(self):
        # 20:40 screen off in a backpack (dark, locked), then the lid closes: the recording stops for 40 minutes
        s = night([((20, 0, 1), (20, 40, 1), 300, 3, False, 2),
                   ((20, 40, 1), (21, 0, 1), 0, 1200, True, 0),
                   ((21, 40, 1), (23, 50, 1), 390, 3, False, 2),   # at home, using it again
                   ((23, 50, 1), (8, 0, 2), 0, 9000, True, 0)])
        r = analyze(s)
        self.assertGreater(r["onset"], ts(23, 50, 1))

    def test_mac_left_locked_at_home_is_not_sleep(self):
        # the sensor starts at 20:00 on a Mac that was locked hours ago: dark and locked, but nobody put it away just now
        s = night([((20, 0, 1), (23, 59, 1), 0, 30000, True, 0)])
        self.assertNotIn("onset", analyze(s))

    def test_daytime_rest_is_not_night_sleep(self):
        s = night([((13, 30, 1), (14, 0, 1), 300, 3, False, 2), ((14, 0, 1), (17, 0, 1), 0, 9000, True, 0)])
        self.assertNotIn("onset", analyze(s))


if __name__ == "__main__":
    unittest.main()
