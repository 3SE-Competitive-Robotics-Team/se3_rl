import argparse
import select
import socket
import threading
from contextlib import suppress
from urllib.parse import urlsplit

BUFFER_SIZE = 65536


def relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 120)
            if not readable:
                return
            for src in readable:
                data = src.recv(BUFFER_SIZE)
                if not data:
                    return
                dst = right if src is left else left
                dst.sendall(data)
    finally:
        for sock in sockets:
            with suppress(OSError):
                sock.close()


def read_headers(client: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = client.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def handle_connect(client: socket.socket, target: str, rest: bytes) -> None:
    host, _, port_text = target.rpartition(":")
    if not host:
        host = target
        port = 443
    else:
        port = int(port_text)
    upstream = socket.create_connection((host, port), timeout=15)
    client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
    if rest:
        upstream.sendall(rest)
    relay(client, upstream)


def handle_http(
    client: socket.socket,
    method: str,
    url: str,
    version: str,
    header_blob: bytes,
) -> None:
    parsed = urlsplit(url)
    if not parsed.hostname:
        client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    lines = header_blob.split(b"\r\n")
    rewritten = [f"{method} {path} {version}".encode()]
    for line in lines[1:]:
        if not line:
            break
        if line.lower().startswith(b"proxy-connection:"):
            continue
        rewritten.append(line)
    request = b"\r\n".join(rewritten) + b"\r\n\r\n"

    upstream = socket.create_connection((parsed.hostname, port), timeout=15)
    upstream.sendall(request)
    relay(client, upstream)


def handle_client(client: socket.socket) -> None:
    try:
        header_blob = read_headers(client)
        if not header_blob:
            client.close()
            return
        head, _, rest = header_blob.partition(b"\r\n\r\n")
        first = head.split(b"\r\n", 1)[0].decode("latin1")
        parts = first.split()
        if len(parts) != 3:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            client.close()
            return
        method, target, version = parts
        if method.upper() == "CONNECT":
            handle_connect(client, target, rest)
        else:
            handle_http(client, method, target, version, header_blob)
    except Exception:
        with suppress(OSError):
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(128)
    print(f"listening on {args.host}:{args.port}", flush=True)
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle_client, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
