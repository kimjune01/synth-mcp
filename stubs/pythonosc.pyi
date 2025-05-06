from typing import Callable, Any, Tuple, Optional

class Dispatcher:
    def __init__(self) -> None: ...
    def map(self, address: str, handler: Callable[..., None]) -> None: ...

class ThreadingOSCUDPServer:
    def __init__(
        self, server_address: Tuple[str, int], dispatcher: Dispatcher
    ) -> None: ...
    def serve_forever(self) -> None: ...
    server_address: Tuple[str, int]

# Module-level exports
osc_server: Any
dispatcher: Any
