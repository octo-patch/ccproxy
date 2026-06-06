"""Probe runtime properties from inside a ccproxy network namespace."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)  # noqa: S603
    except Exception as exc:
        return f"ERROR: {exc}"
    if result.returncode != 0:
        return (result.stderr or result.stdout).strip()
    return result.stdout.strip()


def _tcp_connect(host: str, port: int, *, family: socket.AddressFamily = socket.AF_UNSPEC) -> bool:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(3.0)
            sock.connect((host, port))
            return True
    except OSError:
        return False


def _dns_lookup(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def probe(proxy_port: int) -> dict[str, Any]:
    resolver_path = Path("/etc/resolv.conf")
    resolver_config = resolver_path.read_text(errors="replace") if resolver_path.exists() else ""

    return {
        "route_table": _run_text(["ip", "route"]),
        "resolver_config": resolver_config,
        "dns_lookup_ok": _dns_lookup("example.com"),
        "public_ipv4_ok": _tcp_connect("1.1.1.1", 443, family=socket.AF_INET),
        "public_ipv6_ok": _tcp_connect("2606:4700:4700::1111", 443, family=socket.AF_INET6),
        "ccproxy_port_ok": _tcp_connect("127.0.0.1", proxy_port, family=socket.AF_INET),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe observed ccproxy namespace behavior.")
    parser.add_argument("--proxy-port", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.proxy_port), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
