"""A real loopback ASGI server/CLI smoke test, in addition to in-process API tests."""

import os
import socket
import subprocess
import sys
import time

import httpx


def test_real_cli_server_and_two_authenticated_clients(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = tmp_path / "api.yaml"
    config.write_text(f"""schema_version: '1'
api:
  port: {port}
tenants:
  - id: alpha
    api_key_env: OFLOW_ALPHA_API_KEY
  - id: beta
    api_key_env: OFLOW_BETA_API_KEY
""")
    environment = {
        **os.environ,
        "OFLOW_ALPHA_API_KEY": "a" * 40,
        "OFLOW_BETA_API_KEY": "b" * 40,
        "OFLOW_API_URL": f"http://127.0.0.1:{port}",
    }
    log_path = tmp_path / "server.log"
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "oflow", "serve", "--config", str(config)],
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        with httpx.Client(
            base_url=environment["OFLOW_API_URL"], timeout=2, trust_env=False
        ) as client:
            for _ in range(100):
                assert process.poll() is None, log_path.read_text(encoding="utf-8")
                try:
                    if client.get("/health/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.05)
            else:
                raise AssertionError("ASGI server did not become ready")
            assert (
                client.post(
                    "/v1/tasks/execute", json={"task": "text.score", "input": {"text": "test"}}
                ).status_code
                == 401
            )
        clients = subprocess.run(
            [sys.executable, "examples/api_clients.py"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert clients.returncode == 0, clients.stderr
        assert "ALPHA" in clients.stdout and "BETA" in clients.stdout
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
        log_file.close()
