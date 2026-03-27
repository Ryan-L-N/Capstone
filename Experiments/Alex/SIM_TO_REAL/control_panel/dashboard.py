"""Lightweight web dashboard for the live training control panel.

Serves a single-page HTML dashboard with sliders and buttons.
Uses only Python stdlib (http.server + json). Zero external dependencies.

Usage:
    python -m control_panel.dashboard --port 6008
    python -m control_panel.dashboard --run /path/to/run/dir --port 6008

Then open http://172.24.254.24:6008 in your browser.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import yaml
    _USE_YAML = True
except ImportError:
    _USE_YAML = False

_DASHBOARD_HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


def _find_latest_run() -> str:
    base = os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl")
    base = os.path.abspath(base)
    candidates = []
    for exp_dir in glob.glob(os.path.join(base, "*")):
        ctrl = os.path.join(exp_dir, "control.yaml")
        if os.path.exists(ctrl):
            candidates.append((os.path.getmtime(ctrl), exp_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _read_control(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = f.read()
    if _USE_YAML:
        return yaml.safe_load(raw) or {}
    return json.loads(raw) if raw.strip() else {}


def _write_control(path, data):
    tmp = path + ".tmp"
    if _USE_YAML:
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(data, indent=2)
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _read_history(log_dir, n=50):
    path = os.path.join(log_dir, "control_panel_changes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.readlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    return entries


class DashboardHandler(BaseHTTPRequestHandler):

    ctrl_path = ""
    log_dir = ""

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404, "dashboard.html not found")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._send_html(_DASHBOARD_HTML)

        elif parsed.path == "/api/state":
            data = _read_control(self.ctrl_path)
            data["history"] = _read_history(self.log_dir, n=30)
            self._send_json(data)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/command":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                command = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            command["timestamp"] = datetime.now().isoformat(timespec="seconds")

            data = _read_control(self.ctrl_path)
            if "pending_commands" not in data:
                data["pending_commands"] = []
            data["pending_commands"].append(command)
            _write_control(self.ctrl_path, data)

            self._send_json({"ok": True, "command": command})

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Live Training Dashboard")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory (auto-detect if omitted)")
    args = parser.parse_args()

    run_dir = args.run or _find_latest_run()
    if not run_dir:
        print("ERROR: No run directory with control.yaml found.")
        print("Start a training run with control panel enabled first.")
        sys.exit(1)

    ctrl_path = os.path.join(run_dir, "control.yaml")
    DashboardHandler.ctrl_path = ctrl_path
    DashboardHandler.log_dir = run_dir

    print(f"Dashboard serving: {run_dir}")
    print(f"Control file:      {ctrl_path}")
    print(f"Open browser:      http://0.0.0.0:{args.port}")
    print(f"Press Ctrl+C to stop.\n")

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
