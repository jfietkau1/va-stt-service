import asyncio
import json
import logging

import websockets
from websockets.asyncio.server import Server, ServerConnection

logger = logging.getLogger(__name__)


class WsServer:
    """WebSocket server that broadcasts STT events to connected clients.

    Accepts a single orchestrator client. If multiple connect, all receive events.
    """

    def __init__(self, host: str, port: int, event_queue: asyncio.Queue):
        self._host = host
        self._port = port
        self._event_queue = event_queue
        self._clients: set[ServerConnection] = set()
        self._server: Server | None = None

    async def start(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info("WebSocket server listening on ws://%s:%d", self._host, self._port)
        asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")

    async def _handler(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)
        try:
            async for _ in websocket:
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    async def _broadcast_loop(self) -> None:
        while True:
            event = await self._event_queue.get()
            message = json.dumps(event)

            if not self._clients:
                logger.debug("No clients connected, dropping event: %s", event.get("type"))
                continue

            disconnected = set()
            for client in self._clients:
                try:
                    await client.send(message)
                except websockets.ConnectionClosed:
                    disconnected.add(client)

            self._clients -= disconnected
