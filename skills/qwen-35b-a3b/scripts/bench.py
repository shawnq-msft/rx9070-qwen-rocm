#!/usr/bin/env python3
"""Quick PP/TG bench against a running llama-server on 127.0.0.1:8765."""
import urllib.request, json, time, sys

URL = "http://127.0.0.1:8765/v1/chat/completions"
PROMPT = "Count from 1 to 50, one number per line, no commentary."
MAX_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 120

req = urllib.request.Request(URL,
    data=json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOK, "temperature": 0.7, "stream": False,
    }).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.loads(urllib.request.urlopen(req, timeout=240).read())
wall = time.time() - t0
t = d.get("timings", {})
u = d.get("usage", {})
print(f"PP={round(t.get('prompt_per_second',0),2)} TG={round(t.get('predicted_per_second',0),2)} "
      f"prompt_tok={u.get('prompt_tokens')} comp_tok={u.get('completion_tokens')} wall={round(wall,2)}s")
