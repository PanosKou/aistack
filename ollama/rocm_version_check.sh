#!/usr/bin/env bash

for tag in 0.30.3-rocm 0.30.2-rocm 0.30.1-rocm 0.30.0-rocm 0.29.0-rocm 0.28.0-rocm 0.27.0-rocm; do
  echo
  echo "===== $tag ====="

  sudo docker pull "ollama/ollama:$tag" >/dev/null 2>&1 || {
    echo "pull failed"
    continue
  }

  sudo docker run --rm --entrypoint /bin/sh "ollama/ollama:$tag" -lc '
    ollama --version || true
    find /usr/lib/ollama -maxdepth 2 -type d | grep rocm || true
    find /usr/lib/ollama -maxdepth 3 -type f | grep -Ei "libggml-hip|libamdhip|libhsa|rocm" | head -30 || true
  ' || echo "run failed"
done
