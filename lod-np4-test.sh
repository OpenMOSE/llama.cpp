#!/bin/bash
export HIP_VISIBLE_DEVICES=0
./build/bin/llama-server -m /home/mose/Projects/llm/Qwen3.6-27B-Q4_K_M.gguf --lod -np 4 -c 32768 -ngl 99 -fa on --port 18089 --host 127.0.0.1 > /tmp/lodsrv.log 2>&1 &
SP=$!
ok=0
for i in $(seq 1 150); do
    sleep 2
    if curl -s http://127.0.0.1:18089/health | grep -q ok; then ok=1; break; fi
done
echo "health=$ok"
if [ "$ok" = 1 ]; then
    for i in 1 2 3 4; do
        curl -s --max-time 150 http://127.0.0.1:18089/completion -d "{\"prompt\":\"Q$i: what is $i plus $i? Answer with the number only. A:\",\"n_predict\":10,\"temperature\":0}" > /tmp/r$i.json &
    done
    wait
    for i in 1 2 3 4; do echo "R$i: $(head -c 120 /tmp/r$i.json)"; done
    echo "--- server log tail ---"
    tail -15 /tmp/lodsrv.log
else
    tail -8 /tmp/lodsrv.log
fi
kill $SP 2>/dev/null
