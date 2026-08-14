"""Score a NIAH probe against llama.cpp, so LoD2 can be compared to the author's numbers.

The probe files store token ids produced by the model's own tokenizer, and the
author's verify_lod2.py scores by feeding ``ids[:answer_start]`` and checking
the generated continuation against ``ids[answer_start:answer_start+answer_len]``.
llama-server accepts a token array as a prompt, so the same protocol works here
without re-tokenizing anything - which matters, because a re-tokenized needle is
not the same needle.

  python niah_llamacpp.py --probe <probe.json> --ctx 32768 --docs 8 \\
      --model /home/mose/Projects/llm/Qwen3.5-2B-BF16.gguf --lod2

The server is started and stopped by this script, so a run cannot accidentally
score against a server left over from a different configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise SystemExit("server did not become ready")


def complete(port: int, ids: list[int], n_predict: int) -> str:
    body = json.dumps({
        "prompt": ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=["niah_single_3", "niah_multikey_2"])
    ap.add_argument("--ubatch", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--lod2", action="store_true")
    ap.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for the server")
    ap.add_argument("--server", default="./build/bin/llama-server")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.probe))["rows"] if r["ctx"] == a.ctx]
    by: dict[str, list] = {}
    for r in rows:
        if r["task"] in a.tasks:
            by.setdefault(r["task"], []).append(r)
    if not by:
        raise SystemExit(f"{a.tasks} @ {a.ctx} not in the probe")

    port = free_port()
    env = dict(os.environ)
    if a.lod2:
        env["LLAMA_LOD2"] = "1"
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = [a.server, "-m", a.model, "-c", str(a.ctx + 512), "-ub", str(a.ubatch),
           "-b", str(a.ubatch), "-ngl", str(a.ngl), "--port", str(port),
           "-np", "1", "--no-warmup"]
    print("[niah] " + " ".join(cmd), flush=True)
    print("[niah] lod2 = " + ("on" if a.lod2 else "off")
          + "".join(f"  {k}" for k in a.env), flush=True)

    log = open("/tmp/mose/lod2/niah-server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    results = {"ctx": a.ctx, "lod2": a.lod2, "env": a.env, "scores": {}}
    try:
        wait_ready(port, proc, timeout=600)
        for task, all_rows in sorted(by.items()):
            rs = all_rows[: a.docs]
            hit = 0
            for i, r in enumerate(rs):
                start, length = r["answer_start"], r["answer_len"]
                got = complete(port, r["ids"][:start], length + 8)
                gold = r["gold_text"].strip()
                ok = gold and gold in got
                hit += bool(ok)
                print(f"    {task} {i}: {'HIT ' if ok else 'MISS'} "
                      f"gold={gold[:40]!r} got={got.strip()[:60]!r}", flush=True)
            score = hit/len(rs)
            results["scores"][task] = {"hit": hit, "docs": len(rs), "score": score}
            print(f"  {task:<18} {hit}/{len(rs)} = {score:.3f}", flush=True)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
        log.close()

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
    print(json.dumps(results["scores"]), flush=True)


if __name__ == "__main__":
    main()
