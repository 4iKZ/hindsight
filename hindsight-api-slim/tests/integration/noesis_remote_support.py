"""Shared SSH support for explicitly enabled Noesis remote tests."""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import threading

EXPECTED_HOST_KEY = "SHA256:fYPgM4a2OY1ZRhdQbx2z2YjiQ9bOMx4zo/c1ewn+WCs"
SSH_ENV_KEYS = (
    "NOESIS_SSH_HOST",
    "NOESIS_SSH_PORT",
    "NOESIS_SSH_USER",
    "NOESIS_SSH_PW",
)


def remote_enabled() -> bool:
    return os.environ.get("NOESIS_REMOTE_TEST") == "1" and all(
        os.environ.get(key) for key in SSH_ENV_KEYS
    )


class SshTunnel:
    """Forward local ports to localhost ports on the pinned SSH host."""

    def __init__(self, remote_ports: tuple[int, ...] = (5432,)) -> None:
        import paramiko

        self.transport = paramiko.Transport(
            (os.environ["NOESIS_SSH_HOST"], int(os.environ["NOESIS_SSH_PORT"]))
        )
        self.transport.start_client(timeout=20)
        key = self.transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(
            hashlib.sha256(key.asbytes()).digest()
        ).decode().rstrip("=")
        if fingerprint != EXPECTED_HOST_KEY:
            self.transport.close()
            raise RuntimeError("remote host key mismatch — refusing to connect")
        self.transport.auth_password(
            username=os.environ["NOESIS_SSH_USER"],
            password=os.environ["NOESIS_SSH_PW"],
        )

        self._stopping = threading.Event()
        self._servers: list[socket.socket] = []
        self.ports: dict[int, int] = {}
        for remote_port in remote_ports:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(8)
            self._servers.append(server)
            self.ports[remote_port] = server.getsockname()[1]
            threading.Thread(
                target=self._serve,
                args=(server, remote_port),
                daemon=True,
            ).start()
        self.port = self.ports.get(5432, next(iter(self.ports.values())))

    def _serve(self, server: socket.socket, remote_port: int) -> None:
        while not self._stopping.is_set():
            try:
                client, _ = server.accept()
            except OSError:
                return
            threading.Thread(
                target=self._forward,
                args=(client, remote_port),
                daemon=True,
            ).start()

    def _forward(self, client: socket.socket, remote_port: int) -> None:
        try:
            channel = self.transport.open_channel(
                "direct-tcpip",
                ("localhost", remote_port),
                client.getsockname(),
            )
        except Exception:
            client.close()
            return

        def pump(src, dst) -> None:
            try:
                while data := src.recv(65536):
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        threading.Thread(target=pump, args=(client, channel), daemon=True).start()
        pump(channel, client)

    def close(self) -> None:
        self._stopping.set()
        for server in self._servers:
            try:
                server.close()
            except OSError:
                pass
        self.transport.close()
