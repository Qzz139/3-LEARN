#!/usr/bin/env python3
# 红外只读测量工具：保存四通道原始读数，与人工尺量结果分别记录。
"""Record EP distance telemetry only; never issue motion or LED commands."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import statistics
import threading
import time

from ir_pick_place_demo import ROOT, configure_network, load_config


# 统计正数有效读数和阈值内最长连续段，辅助比较尺量距离与红外原始值。
def summarize(values, threshold_mm):
    valid = [v for v in values if 0 < v <= 10000]
    longest = streak = 0
    for value in values:
        streak = streak + 1 if 0 < value <= threshold_mm else 0
        longest = max(longest, streak)
    return {
        "samples": len(values), "valid_samples": len(valid),
        "invalid_samples": len(values) - len(valid),
        "min_mm": min(valid) if valid else None,
        "median_mm": statistics.median(valid) if valid else None,
        "max_mm": max(valid) if valid else None,
        "stdev_mm": round(statistics.pstdev(valid), 2) if valid else None,
        "threshold_mm": threshold_mm,
        "within_threshold_samples": sum(0 < v <= threshold_mm for v in values),
        "longest_within_threshold_streak": longest,
    }


# 只订阅测距并保存原始样本和统计报告，不发送底盘、舵机、抓夹或LED命令。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "ir_pick_place.json")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--reference-mm", type=float, help="manually measured surface distance")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 300:
        parser.error("seconds must be within [1, 300]")
    if args.reference_mm is not None and (not math.isfinite(args.reference_mm) or args.reference_mm <= 0):
        parser.error("reference-mm must be positive")
    cfg = load_config(args.config)
    (ROOT / "work").mkdir(exist_ok=True)
    lock_file = (ROOT / "work" / "real-control.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, "another EP control/recording process is running\n")
    from robomaster import config, robot
    configure_network(cfg["connection"], config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "work" / "ir-distance" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    bot = robot.Robot()
    subscribed = False
    samples, lock = [], threading.Lock()
    started = time.monotonic()
    error = None
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as stream:
        # 回调中先复制四通道数组，再在锁内逐行保存，避免SDK复用数组污染记录。
        def receive(data):
            raw = copy.deepcopy(data)
            if len(raw) != 4:
                return
            item = {"elapsed_s": round(time.monotonic() - started, 4),
                    "distance_mm": list(raw)}
            with lock:
                samples.append(item)
                stream.write(json.dumps(item) + "\n")
                stream.flush()

        try:
            if not bot.initialize(conn_type=cfg["connection"]["conn_type"], proto_type="udp"):
                raise RuntimeError("SDK initialization failed")
            subscribed = bool(bot.sensor.sub_distance(freq=20, callback=receive))
            if not subscribed:
                raise RuntimeError("distance subscription rejected")
            print("READ ONLY: label={}, reference_mm={}, duration={}s".format(
                args.label, args.reference_mm, args.seconds), flush=True)
            deadline = time.monotonic() + args.seconds
            cursor = 0
            while time.monotonic() < deadline:
                time.sleep(min(1.0, max(0, deadline - time.monotonic())))
                with lock:
                    recent = samples[cursor:]
                    cursor = len(samples)
                slot = cfg["infrared"]["feedback_slot"]
                print(json.dumps({"slot": slot,
                                  "latest_all_slots_mm": recent[-1]["distance_mm"] if recent else None,
                                  "last_second": summarize([r["distance_mm"][slot] for r in recent], cfg["infrared"]["threshold_mm"])}), flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            if subscribed:
                try:
                    bot.sensor.unsub_distance()
                except Exception:
                    pass
            try:
                bot.close()
            except Exception:
                pass
            with lock:
                report = {"label": args.label, "reference_mm": args.reference_mm,
                          "motion_commands_sent": False, "error": error,
                          "configured_slot": cfg["infrared"]["feedback_slot"],
                          "threshold_mm": cfg["infrared"]["threshold_mm"],
                          "max_callback_gap_s": round(max(
                              (b["elapsed_s"] - a["elapsed_s"] for a, b in zip(samples, samples[1:])),
                              default=0.0), 4),
                          "slots": [summarize([r["distance_mm"][i] for r in samples], cfg["infrared"]["threshold_mm"])
                                    for i in range(4)]}
            (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("SUMMARY " + json.dumps(report), flush=True)
            print("RECORD " + str(directory), flush=True)
    return 1 if error or not samples else 0


if __name__ == "__main__":
    raise SystemExit(main())
