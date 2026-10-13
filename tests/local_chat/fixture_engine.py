"""Deterministic JSON-lines subprocess used by EngineSession integration tests."""
import json
import subprocess
import sys
import time


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    mode = next((a.split("=", 1)[1] for a in args if a.startswith("--mode=")), "normal")
    context = int(next((a.split("=", 1)[1] for a in args if a.startswith("--max-context=")), "4096"))
    if mode == "capabilities_mismatch":
        context += 1
    emit({"type": "ready", "protocol": "silicon-chat-v1", "max_context": context,
          "streaming": mode != "nonstream"})

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
        except Exception:
            if mode == "invalid_line":
                sys.stdout.write("this is not json\n")
                sys.stdout.flush()
                continue
            continue
        if mode == "invalid_line":
            sys.stdout.write("this is not json\n")
            sys.stdout.flush()
            continue
        rid = request.get("request_id", "")
        if request.get("reset") is not True:
            emit({"type": "error", "request_id": rid, "message": "request reset flag was not true"})
            continue
        if mode == "mismatched_request_id":
            rid = "unexpected-request"
        if mode == "slow_timeout":
            time.sleep(5)
            continue
        if mode == "child_process":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            emit({"type": "delta", "request_id": rid,
                  "text": str(child.pid), "generated_tokens": 1})
            time.sleep(60)
            continue
        if mode == "context_rejection":
            emit({"type": "status", "request_id": rid, "stage": "prefill"})
            messages = request.get("messages", [])
            if any(len(message.get("content", "")) > context for message in messages):
                emit({"type": "error", "request_id": rid,
                      "message": "fixture context limit exceeded"})
                continue
        if mode == "slow":
            emit({"type": "status", "request_id": rid, "stage": "decode"})
            i = 0
            while True:
                emit({"type": "delta", "request_id": rid, "text": "x", "generated_tokens": i + 1})
                i += 1
                time.sleep(0.05)
        if mode == "oversized_event":
            sys.stdout.write('{"type":"' + ("x" * (1024 * 1024 + 10)) + "}\n")
            sys.stdout.flush()
            continue

        if mode != "context_rejection":
            emit({"type": "status", "request_id": rid, "stage": "prefill"})
        emit({"type": "status", "request_id": rid, "stage": "decode"})
        text = "Ciao 🌱 — café"
        if mode == "response_mismatch":
            emit({"type": "delta", "request_id": rid, "text": "parziale", "generated_tokens": 1})
            result = "testo diverso"
        elif mode == "over_limit":
            emit({"type": "delta", "request_id": rid, "text": "troppo", "generated_tokens": request["max_new_tokens"] + 1})
            continue
        elif mode == "bad_metrics":
            result = text
        else:
            result = text

        if mode != "nonstream" and mode != "response_mismatch" and mode != "bad_metrics":
            emit({"type": "delta", "request_id": rid, "text": text, "generated_tokens": 4})
        if mode == "normal":
            time.sleep(0.1)
        stats = {"generated_tokens": 4, "decode_seconds": 0.25, "prefill_seconds": 0.1,
                 "load_seconds": 0.0, "working_set_bytes": 1024}
        if mode == "bad_metrics":
            stats["decode_seconds"] = -1
        emit({"type": "completed", "request_id": rid, "text": result,
              "statistics": stats, "termination": "eos"})


if __name__ == "__main__":
    main()
