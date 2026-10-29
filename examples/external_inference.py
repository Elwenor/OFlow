"""Self-contained trusted fixed-URL HTTP backend using a local test inference server."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from oflow import OFlowEngine, TaskRequest
from oflow.config import EngineConfig


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        words = body["input"]["text"].split()
        encoded = json.dumps(
            {
                "score": len(set(words)) / max(1, len(words)),
                "words": len(words),
                "method": "local_test_server",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


async def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = EngineConfig.model_validate(
        {
            "backends": [
                {
                    "id": "remote",
                    "kind": "http",
                    "options": {
                        "url": f"http://127.0.0.1:{server.server_port}/infer",
                        "allow_insecure_localhost": True,
                    },
                }
            ]
        }
    )
    try:
        async with OFlowEngine(config) as engine:
            result = await engine.execute(
                TaskRequest(task="text.score", input={"text": "hello remote"})
            )
            assert not result.error
            print(result.model_dump_json(indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


if __name__ == "__main__":
    asyncio.run(main())
