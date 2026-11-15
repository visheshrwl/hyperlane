"""Application-level round trip for one activation payload over TCP.

python3 spike/tcp_rtt.py serve HOST PORT
python3 spike/tcp_rtt.py probe HOST PORT [--bytes 8192] [--count 300]
"""

import argparse
import socket
import statistics
import time


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def serve(host: str, port: int) -> None:
    with socket.create_server((host, port)) as server:
        while True:
            conn, _ = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    while True:
                        size = int.from_bytes(recv_exact(conn, 4), "little")
                        conn.sendall(recv_exact(conn, size))
                except ConnectionError:
                    pass


def probe(host: str, port: int, size: int, count: int) -> None:
    payload = bytes(size)
    with socket.create_connection((host, port)) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        samples = []
        for _ in range(count):
            start = time.perf_counter()
            sock.sendall(len(payload).to_bytes(4, "little") + payload)
            recv_exact(sock, size)
            samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    pct = lambda p: samples[min(len(samples) - 1, int(p / 100 * len(samples)))]  # noqa: E731
    print(
        f"{host}:{port} {size} B x{count}: p50 {pct(50):.1f} ms  p90 {pct(90):.1f} ms  "
        f"p99 {pct(99):.1f} ms  min {samples[0]:.1f} ms  mean {statistics.mean(samples):.1f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "probe"])
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("--bytes", type=int, default=8192)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    if args.mode == "serve":
        serve(args.host, args.port)
    else:
        probe(args.host, args.port, args.bytes, args.count)
