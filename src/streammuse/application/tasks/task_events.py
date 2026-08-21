"""Configuration and no-op implementation for interactive task observers."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

from streammuse.domain.tasks import TaskViewEvent


@dataclass(frozen=True)
class TaskWebConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8002
    allow_remote: bool = False
    queue_capacity: int = 256
    server_start_timeout_s: float = 5.0
    flush_timeout_s: float = 0.5
    shutdown_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        host = str(self.host).strip()
        if not host:
            raise ValueError("web host must not be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("web port must be between 1 and 65535")
        if int(self.queue_capacity) <= 0:
            raise ValueError("web queue capacity must be > 0")
        for name in (
            "server_start_timeout_s",
            "flush_timeout_s",
            "shutdown_timeout_s",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.enabled and not self.allow_remote and not _is_loopback(host):
            raise ValueError(
                "non-loopback --web-host requires --web-allow-remote"
            )


class NullTaskEventSink:
    """Default observer that leaves interactive gameplay unchanged."""

    def emit(self, event: TaskViewEvent) -> None:
        del event

    def close(self) -> None:
        return None


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
