"""services/llm.py against a fake Ollama server (no live model here): the streamed response is
reassembled, timings and options reach the server, a server without JSON-schema `format`
support falls back to plain JSON mode, and a cancelled call stops reading the stream."""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import llm

requests_seen = []
mode = {"reject_schema": False, "slow": False}


class FakeOllama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # chunked streaming, like Ollama

    def _chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests_seen.append(body)
        if mode["reject_schema"] and isinstance(body["format"], dict):
            payload = b'{"error":"invalid format: expected \\"json\\" or a JSON schema"}'
            self.send_response(400)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        text = '{"state": "consistent", "plausibility": "adequate", "reasoning": "ok", "new_claim_text": null}'
        try:
            for piece in (text[:20], text[20:50], text[50:]):
                self._chunk((json.dumps({"response": piece, "done": False}) + "\n").encode())
                if mode["slow"]:
                    time.sleep(0.3)
            final = {"response": "", "done": True, "prompt_eval_count": 42, "prompt_eval_duration": 5_000_000,
                     "eval_count": 30, "eval_duration": 90_000_000, "load_duration": 1_000_000}
            self._chunk((json.dumps(final) + "\n").encode())
            self._chunk(b"")
        except ConnectionError:
            pass  # the client cancelled

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        stats = {}
        data = llm.call_agent(host, "m", "p", validate=llm.validate_analyst_response,
                              schema=llm.ANALYST_SCHEMA, num_ctx=8192, stats=stats)
        assert data["state"] == "consistent", data
        sent = requests_seen[-1]
        assert sent["stream"] is True and sent["format"] == llm.ANALYST_SCHEMA, sent
        assert sent["options"] == {"num_predict": 256, "num_ctx": 8192}, sent["options"]
        assert stats["prompt_tokens"] == 42 and stats["output_tokens"] == 30 and stats["output_ms"] == 90, stats

        # Ollama < 0.5 rejects a schema: one fallback to "json", then plain JSON mode from then on.
        mode["reject_schema"] = True
        llm.call_agent(host, "m", "p", validate=llm.validate_analyst_response, schema=llm.ANALYST_SCHEMA)
        assert requests_seen[-1]["format"] == "json" and llm._schema_supported is False
        mode["reject_schema"] = False
        llm._schema_supported = True

        # Cancel: the call raises Cancelled instead of waiting for the full answer.
        mode["slow"] = True
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        start = time.time()
        try:
            llm.call_agent(host, "m", "p", schema=llm.ANALYST_SCHEMA, cancel=cancel)
            assert False, "should have been cancelled"
        except llm.Cancelled:
            pass
        assert time.time() - start < 0.8, time.time() - start
        mode["slow"] = False
    finally:
        server.shutdown()
    print("test_llm_client.py passed")


if __name__ == "__main__":
    main()
