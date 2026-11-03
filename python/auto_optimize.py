"""Run experiments unattended: pick the next one, train it, score it on validation, compare, decide what to try next, repeat until the deadline.

It keeps working if the Claude session ends. Test clips are never scored here. Add work at any time by appending to experiments/queue.json.

  python auto_optimize.py --stop-at 2026-11-03T05:15:00+05:30

Files it keeps: experiments/exp_NNN/, experiments/queue.json, experiments/leaderboard.csv, experiments/BEST_MODEL.json,
research/PROGRESS.md, research/RESEARCH_LOG.md, logs/auto_optimize.log
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

EXP, QUEUE, BEST = "experiments", "experiments/queue.json", "experiments/BEST_MODEL.json"
MAX_RETRIES, MIN_GAIN, MAX_PARAMS, DISK_BUDGET_GB = 2, 0.003, 2_500_000, 10
BASE_VAL = {"name": "final_v2 (deployed)", "dir": "model_out/final_v2"}
SINGLE_REF = {"name": "f_long_s0 (single BiLSTM, 180 epochs)", "dir": "model_out/f_long_s0"}


def now():
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def say(*a):
    print(now(), *a, flush=True)


def read_json(p, default):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def write_json(p, v):
    tmp = p + ".tmp"
    json.dump(v, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def signature(c):
    keep = {k: v for k, v in c.items() if k not in ("threads", "max_minutes")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def paired_diff(a, b, n=2000):
    """Paired bootstrap of top-1(a) - top-1(b) on the validation clips both runs scored. Returns mean and 95% interval in fractions."""
    def load(d):
        y = np.load(f"{d}/val_y.npy")
        return dict(zip(np.load(f"{d}/val_videos.npy").tolist(), (np.load(f"{d}/val_probs.npy").argmax(1) == y).tolist()))
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    if not common:
        return None
    d = np.array([A[v] for v in common], float) - np.array([B[v] for v in common], float)
    r = np.random.RandomState(0).randint(0, len(d), (n, len(d)))
    m = d[r].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(common)}


def battery_low():
    """True when on battery below 15%: starting long work then risks losing it."""
    try:
        o = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        pct = int(re.search(r"(\d+)%", o).group(1))
        return "Battery Power" in o and pct < 15
    except Exception:
        return False


def disk_used_gb():
    """This run's additions, against the 10 GB budget."""
    tot = 0
    for p in [EXP, os.path.expanduser("~/signo_data/asl_citizen/raw"), os.path.expanduser("~/signo_data/asl_citizen/fs"), os.path.expanduser("~/signo_data/asl_citizen/fresh_test_videos")]:
        for d, _, fs in os.walk(p):
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
    return tot / 1e9


def memory_free_pct():
    try:
        return int(re.search(r"(\d+)%", subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.splitlines()[-1]).group(1))
    except Exception:
        return 100


def extraction_pids():
    o = subprocess.run(["pgrep", "-f", "extract_raw.py|multiprocessing.spawn|multiprocessing.resource_tracker"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in o if int(p) != os.getpid()]


def extraction_running():
    """True when extraction is competing for the CPU. Paused workers do not count."""
    for pid in extraction_pids():
        st = subprocess.run(["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if st and not st.startswith("T"):
            return True
    return False


def resume_extraction():
    """If extraction was paused to free the CPU and there is nothing to train, let it carry on."""
    for pid in extraction_pids():
        try:
            os.kill(pid, 18)  # SIGCONT
        except OSError:
            pass


def finished():
    out = []
    for p in sorted(glob.glob(f"{EXP}/exp_*/result.json")):
        out.append(read_json(p, None))
    return [r for r in out if r]


def next_id():
    n = [int(m.group(1)) for d in glob.glob(f"{EXP}/exp_*") if (m := re.search(r"exp_(\d+)", d))]
    return max(n, default=0) + 1


def enqueue(q, name, group, hypothesis, config, priority=5, source="controller", reference=None):
    done = {r["signature"] for r in finished()} | {i["signature"] for i in q if "signature" in i} | {signature(i["config"]) for i in q}
    s = signature(config)
    if s in done:
        return False
    q.append({"name": name, "group": group, "hypothesis": hypothesis, "config": config, "priority": priority, "source": source, "signature": s, "status": "pending", "reference": reference})
    return True


def decide_next(r, q):
    """The research rules. Each finished experiment is read for its most likely bottleneck, and at most one follow-up per rule is queued."""
    if r["config"].get("train_limit"):  # partial-data gates are for direction only
        return []
    res = finished()
    best = max((x for x in res if not x["config"].get("train_limit")), key=lambda x: x["metrics"]["top1"], default=r)
    c, m, added = r["config"], r["metrics"], []
    near_best = m["top1"] >= best["metrics"]["top1"] - 0.01
    gen = r.get("generation", 0)
    def add(tag, why, new, pr=6):
        if gen < 3 and enqueue(q, f"{r['name']}+{tag}", r["group"], why, {**c, **new}, pr, reference=r["id"]):
            q[-1]["generation"] = gen + 1
            added.append(f"{r['name']}+{tag}: {why}")
    if near_best and m["gap"] > 0.15:  # memorising training signers
        if not c.get("mixup"):
            add("mixup", f"train-validation gap is {m['gap']:.1%}, so blending clips may reduce memorising", {"mixup": 0.5})
        elif c.get("dropout", 0.4) < 0.6:
            add("dropout", f"gap still {m['gap']:.1%} with mixup, so raise dropout", {"dropout": c.get("dropout", 0.4) + 0.1})
    if near_best and m["gap"] < 0.06:  # no longer memorising, so it may lack capacity or time
        add("longer", f"gap is only {m['gap']:.1%}, so more epochs and capacity may help", {"epochs": int(c.get("epochs", 100) * 1.5), "units": int(c.get("units", 128) * 1.25)})
    if near_best and m["best_epoch"] > 0.9 * m["epochs_run"]:  # still improving at the end
        add("moretime", f"best epoch {m['best_epoch']} of {m['epochs_run']}, so it had not finished learning", {"epochs": int(c.get("epochs", 100) * 1.6)})
    if near_best and m["top5"] - m["top1"] > 0.12 and not c.get("smoothing"):  # right answer in the shortlist, wrong one on top
        add("smooth", f"top-5 is {m['top5'] - m['top1']:.1%} above top-1, so soften targets to separate close signs", {"smoothing": 0.2})
    if m["top1"] >= best["metrics"]["top1"] - 0.002 and not any(x["config"].get("seed") != c.get("seed", 0) and signature({**x["config"], "seed": c.get("seed", 0)}) == signature(c) for x in res):
        add(f"seed{c.get('seed', 0) + 1}", "repeat with a new seed to see whether this result is stable", {"seed": c.get("seed", 0) + 1}, 7)
    return added


def update_best(r):
    """Keep the best model by validation top-1 and macro F1. The previous best is preserved, never overwritten."""
    if r["config"].get("train_limit") or r["metrics"]["params"] > MAX_PARAMS:
        return None
    b = read_json(BEST, None)
    m = r["metrics"]
    if b and not (m["top1"] >= b["metrics"]["top1"] + MIN_GAIN and m["macro_f1"] >= b["metrics"]["macro_f1"] - 0.002):
        return None
    why = f"validation top-1 {m['top1']:.4f} and macro F1 {m['macro_f1']:.4f}" + (f" beat {b['id']} ({b['metrics']['top1']:.4f}, {b['metrics']['macro_f1']:.4f})" if b else "")
    v = {"id": r["id"], "name": r["name"], "dir": r["dir"], "metrics": m, "config": r["config"], "selected_at": now(), "why": why, "previous": (b or {}).get("id"), "selected_on": "validation only"}
    write_json(BEST, v)
    return why


def prune(r):
    """Drop big checkpoints from runs that are not close to the best. The numbers and predictions stay."""
    res = [x for x in finished() if not x["config"].get("train_limit")]
    tops = sorted((x["metrics"]["top1"] for x in res), reverse=True)
    cut = tops[min(7, len(tops) - 1)] if tops else 0
    for x in finished():
        p = f"{x['dir']}/model.keras"
        if os.path.exists(p) and (x["metrics"]["top1"] < cut or x["config"].get("train_limit")):
            os.remove(p)


def log_result(r, ref, added):
    m = r["metrics"]
    lines = [f"\n## {r['id']}: {r['name']}\n", f"- Started {r['started']}, finished {r['finished']}, took {m['train_seconds'] / 60:.0f} minutes",
             f"- Hypothesis: {r['hypothesis']}", f"- Change: `{json.dumps(r['config'], sort_keys=True)}`", f"- Queued by: {r['source']}",
             f"- Validation: top-1 {m['top1']:.2%}, top-5 {m['top5']:.2%}, macro F1 {m['macro_f1']:.2%}, on {m['n_val']} clips. Clean training accuracy {m['train_top1_clean']:.2%}, gap {m['gap']:.1%}. {m['params']:,} parameters"]
    if ref:
        lines.append(f"- Against {ref['name']}: {ref['diff']:+.2%} (95% interval {ref['lo']:+.2%} to {ref['hi']:+.2%}, {ref['n']} shared clips). " + ("Real gain." if ref["lo"] > 0 else "Real loss." if ref["hi"] < 0 else "Within noise."))
    lines.append("- Next, from this result: " + ("; ".join(added) if added else "nothing new queued"))
    open("research/RESEARCH_LOG.md", "a").write("\n".join(lines) + "\n")


def write_board():
    res = finished()
    cols = ["id", "name", "group", "features", "arch", "seed", "epochs", "partial_train", "top1", "top5", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "train_top1_clean", "gap", "best_epoch", "best_val_loss", "params", "train_minutes"]
    with open(f"{EXP}/leaderboard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(res, key=lambda x: -x["metrics"]["top1"]):
            m, c = r["metrics"], r["config"]
            w.writerow([r["id"], r["name"], r["group"], c.get("features", "base"), c["arch"], c.get("seed", 0), c.get("epochs"), c.get("train_limit", "")] + [round(m[k], 4) for k in cols[8:17]] + [m["best_epoch"], round(m["best_val_loss"], 4), m["params"], round(m["train_seconds"] / 60, 1)])


def write_progress(running, q, stop_at, baseline):
    res = sorted(finished(), key=lambda x: -x["metrics"]["top1"])
    b = read_json(BEST, None)
    left = (stop_at - time.time()) / 3600
    o = [f"# Progress\n\nUpdated {now()}. New experiments stop in {left:.1f} hours. Disk used by this run: {disk_used_gb():.2f} GB of {DISK_BUDGET_GB} GB.\n", f"Baseline ({baseline['name']}): validation top-1 {baseline['top1']:.2%}, macro F1 {baseline['macro_f1']:.2%}.\n"]
    o.append(f"Best so far: **{b['id']} {b['name']}**, top-1 {b['metrics']['top1']:.2%}, macro F1 {b['metrics']['macro_f1']:.2%}.\n" if b else "No best model selected yet.\n")
    o.append("## Running now\n" + ("\n".join(f"- {j['id']} {j['item']['name']}, last line: `{j['last']}`" for j in running) or "- nothing") + "\n")
    pend = sorted((i for i in q if i["status"] == "pending"), key=lambda i: -i["priority"])
    o.append(f"## Waiting ({len(pend)})\n" + ("\n".join(f"- {i['name']} (priority {i['priority']})" for i in pend[:15]) or "- nothing") + "\n")
    failed = [i for i in q if i["status"] == "failed"]
    if failed:
        o.append("## Failed\n" + "\n".join(f"- {i['name']}: {i.get('error', '')}" for i in failed) + "\n")
    o.append("## Finished, best first\n\n| id | name | features | top-1 | top-5 | macro F1 | gap | params | partial data |\n|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in res:
        m, c = r["metrics"], r["config"]
        o.append(f"| {r['id']} | {r['name']} | {c.get('features', 'base')} | {m['top1']:.2%} | {m['top5']:.2%} | {m['macro_f1']:.2%} | {m['gap']:.1%} | {m['params']:,} | {c.get('train_limit', '')} |")
    tmp = "research/PROGRESS.md.tmp"
    open(tmp, "w").write("\n".join(o) + "\n")
    os.replace(tmp, "research/PROGRESS.md")


def fix_for_error(item, text):
    """Read a failed run's error and change what it points to. Returns a description, or None if there is nothing to change."""
    c = item["config"]
    if re.search(r"Resource exhausted|OOM|MemoryError|Killed", text) and c.get("batch", 64) > 32:
        c["batch"] = c.get("batch", 64) // 2
        return "out of memory, halved the batch"
    if re.search(r"nan|NaN", text) and c.get("lr", 2e-3) > 3e-4:
        c["lr"] = c.get("lr", 2e-3) / 2
        return "numbers blew up, halved the learning rate"
    if "No such file" in text and c.get("features", "base") != "base":
        return "feature file not built yet"
    return None


class Adopted:
    """A training run started by an earlier controller, which this one takes over after a restart."""

    def __init__(self, pid, d):
        self.pid, self.d = pid, d

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return 0 if os.path.exists(f"{self.d}/metrics.json") else 1


def adopt(q):
    jobs = []
    for i in q:
        if i["status"] != "running":
            continue
        d = f"{EXP}/{i['id']}"
        o = subprocess.run(["pgrep", "-f", f"train_exp.py {d}/config.json"], capture_output=True, text=True).stdout.split()
        if o:
            jobs.append({"id": i["id"], "dir": d, "item": i, "proc": Adopted(int(o[0]), d), "log": f"{d}/train.log", "started": dt.datetime.fromtimestamp(os.path.getmtime(f"{d}/config.json")).astimezone().strftime("%Y-%m-%d %H:%M:%S"), "last": "adopted"})
            say("adopted", i["id"], i["name"])
        else:
            i["status"], i["id"] = "pending", None
            i.pop("id")
            say("re-queued", i["name"], "its process is gone")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", required=True, help="no new experiments after this time")
    ap.add_argument("--run-until", default="", help="every run must end by this time. Defaults to --stop-at")
    ap.add_argument("--slots", type=int, default=3, help="runs at once when extraction has finished")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    stop_at = dt.datetime.fromisoformat(a.stop_at).timestamp()
    os.makedirs(EXP, exist_ok=True)
    p = np.load(f"{BASE_VAL['dir']}/val_probs.npy")
    bm = read_json("research/baseline.json", {})
    baseline = {"name": BASE_VAL["name"], "top1": bm.get("top1", 0), "macro_f1": bm.get("macro_f1", 0)}
    run_until = dt.datetime.fromisoformat(a.run_until).timestamp() if a.run_until else stop_at
    q = read_json(QUEUE, [])
    running = adopt(q)
    write_json(QUEUE, q)
    while True:
        q = read_json(QUEUE, [])
        # finish
        for j in list(running):
            code = j["proc"].poll()
            try:
                tail = open(j["log"]).read().strip().splitlines()[-1][:120]
                j["last"] = tail
            except (OSError, IndexError):
                pass
            if code is None:
                continue
            running.remove(j)
            item = next(i for i in q if i.get("id") == j["id"])
            if code == 0 and os.path.exists(f"{j['dir']}/metrics.json"):
                m = read_json(f"{j['dir']}/metrics.json", {})
                r = {"id": j["id"], "name": item["name"], "group": item["group"], "hypothesis": item["hypothesis"], "config": item["config"], "dir": j["dir"], "metrics": m, "signature": item["signature"], "source": item["source"], "started": j["started"], "finished": now(), "generation": item.get("generation", 0), "reference": item.get("reference")}
                write_json(f"{j['dir']}/result.json", r)
                item["status"] = "done"
                rd = None
                for x in finished():
                    if x["id"] == item.get("reference"):
                        rd = x
                refdir, refname = (rd["dir"], rd["name"]) if rd else (SINGLE_REF["dir"], SINGLE_REF["name"])
                d = paired_diff(j["dir"], refdir) if os.path.exists(f"{refdir}/val_y.npy") else None
                if d:
                    d["name"] = refname
                added = decide_next(r, q)
                why = update_best(r)
                log_result(r, d, added)
                if why:
                    open("research/DECISIONS.md", "a").write(f"\n- {now()}: best model is now {r['id']} {r['name']}: {why}. Chosen on validation only.\n")
                prune(r)
                write_board()
                say("finished", j["id"], item["name"], f"top1 {m['top1']:.4f} f1 {m['macro_f1']:.4f} gap {m['gap']:.3f}", "BEST" if why else "", "| queued:", added)
            else:
                text = open(j["log"]).read()[-4000:]
                item["tries"] = item.get("tries", 0) + 1
                fix = fix_for_error(item, text)
                item["error"] = text.strip().splitlines()[-1][:200] if text.strip() else f"exit {code}"
                item["status"] = "pending" if item["tries"] <= MAX_RETRIES and (fix or item["tries"] == 1) else "failed"
                say("FAILED", j["id"], item["name"], "tries", item["tries"], "fix:", fix, "|", item["error"])
                item.pop("id", None)
                shutil.move(j["dir"], f"{EXP}/_failed_{os.path.basename(j['dir'])}_{int(time.time())}")
            write_json(QUEUE, q)
        # start
        extracting = extraction_running()
        slots = 1 if extracting else a.slots
        threads = 4 if extracting else a.threads
        over = time.time() > stop_at
        hold = None
        if battery_low():
            hold = "battery below 15% and not plugged in"
        elif disk_used_gb() > DISK_BUDGET_GB - 0.5:
            hold = "storage budget nearly used"
        elif memory_free_pct() < 12:
            hold = "memory is short"
        while not over and not hold and len(running) < slots:
            pend = sorted((i for i in q if i["status"] == "pending" and (i["config"].get("features", "base") == "base" or os.path.exists(os.path.expanduser(f"~/signo_data/asl_citizen/fs/{i['config']['features']}.npz")))), key=lambda i: -i["priority"])
            if not pend:
                break
            item = pend[0]
            eid = f"exp_{next_id():03d}"
            d = f"{EXP}/{eid}"
            os.makedirs(d)
            item["config"]["threads"] = threads
            item["config"]["max_minutes"] = min(item["config"].get("max_minutes", 240), max(10, (run_until - time.time()) / 60))
            item["id"], item["status"] = eid, "running"
            item["signature"] = signature(item["config"])
            write_json(f"{d}/config.json", {**item["config"], "name": item["name"], "group": item["group"], "hypothesis": item["hypothesis"], "dataset": "ASL Citizen official splits, 1,000 signs", "train_split": "official train", "val_split": "official val"})
            log = f"{d}/train.log"
            env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
            proc = subprocess.Popen([sys.executable, "python/train_exp.py", f"{d}/config.json"], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
            running.append({"id": eid, "dir": d, "item": item, "proc": proc, "log": log, "started": now(), "last": "starting"})
            write_json(QUEUE, q)
            say("started", eid, item["name"], "threads", threads)
        if hold:
            say("holding:", hold)
        write_progress(running, q, stop_at, baseline)
        if over and not running:
            say("deadline reached, stopping")
            break
        if not running and not any(i["status"] == "pending" for i in q):
            resume_extraction()
        time.sleep(30)


if __name__ == "__main__":
    main()
