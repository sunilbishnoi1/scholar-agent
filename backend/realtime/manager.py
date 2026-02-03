# WebSocket Connection Manager for Real-Time Updates
# Manages WebSocket connections and broadcasts agent progress to connected clients

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""

    websocket: WebSocket
    user_id: str
    connected_at: datetime
    project_ids: set[str]


class ConnectionManager:
    """
    Manages WebSocket connections for real-time agent updates.

    Features:
    - Per-project subscriptions
    - User authentication integration
    - Graceful disconnection handling
    - Broadcast to all subscribers of a project

    Usage:
        manager = ConnectionManager()

        @app.websocket("/ws/projects/{project_id}/stream")
        async def stream_updates(websocket: WebSocket, project_id: str):
            await manager.connect(websocket, user_id, project_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming messages if needed
            except WebSocketDisconnect:
                manager.disconnect(websocket)
    """

    def __init__(self):
        # Map of project_id -> set of websockets subscribed to it
        self._project_connections: dict[str, set[WebSocket]] = {}

        # Map of websocket -> connection info
        self._connection_info: dict[WebSocket, ConnectionInfo] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Statistics
        self._total_connections = 0
        self._total_messages_sent = 0

        # Redis listener task
        self._redis_listener_task: asyncio.Task | None = None
        self._should_listen = False

    async def connect(self, websocket: WebSocket, user_id: str, project_id: str) -> bool:
        """
        Accept a WebSocket connection and subscribe to a project.

        Args:
            websocket: The WebSocket connection
            user_id: The authenticated user's ID
            project_id: The project to subscribe to

        Returns:
            True if connection was established successfully
        """
        try:
            # Accept the WebSocket connection first (critical for race condition prevention)
            await websocket.accept()

            # Small delay to ensure connection is fully established
            await asyncio.sleep(0.01)

            async with self._lock:
                # Track connection info
                if websocket in self._connection_info:
                    # Already connected, just add project subscription
                    self._connection_info[websocket].project_ids.add(project_id)
                else:
                    # New connection
                    self._connection_info[websocket] = ConnectionInfo(
                        websocket=websocket,
                        user_id=user_id,
                        connected_at=datetime.utcnow(),
                        project_ids={project_id},
                    )
                    self._total_connections += 1

                # Subscribe to project
                if project_id not in self._project_connections:
                    self._project_connections[project_id] = set()
                self._project_connections[project_id].add(websocket)

            logger.info(
                f"WebSocket connected: user={user_id}, project={project_id}, "
                f"total_connections={len(self._connection_info)}"
            )

            # Send connection confirmation
            await self._send_to_websocket(
                websocket,
                {
                    "type": "connected",
                    "project_id": project_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            return False

    async def disconnect(self, websocket: WebSocket):
        """
        Clean up when a WebSocket disconnects.

        Args:
            websocket: The disconnected WebSocket
        """
        async with self._lock:
            if websocket not in self._connection_info:
                return

            info = self._connection_info[websocket]

            # Remove from all project subscriptions
            for project_id in info.project_ids:
                if project_id in self._project_connections:
                    self._project_connections[project_id].discard(websocket)
                    # Clean up empty sets
                    if not self._project_connections[project_id]:
                        del self._project_connections[project_id]

            # Remove connection info
            del self._connection_info[websocket]

            logger.info(
                f"WebSocket disconnected: user={info.user_id}, "
                f"remaining_connections={len(self._connection_info)}"
            )

    async def subscribe_to_project(self, websocket: WebSocket, project_id: str):
        """Subscribe an existing connection to an additional project."""
        async with self._lock:
            if websocket not in self._connection_info:
                return False

            self._connection_info[websocket].project_ids.add(project_id)

            if project_id not in self._project_connections:
                self._project_connections[project_id] = set()
            self._project_connections[project_id].add(websocket)

            return True

    async def unsubscribe_from_project(self, websocket: WebSocket, project_id: str):
        """Unsubscribe a connection from a project."""
        async with self._lock:
            if websocket in self._connection_info:
                self._connection_info[websocket].project_ids.discard(project_id)

            if project_id in self._project_connections:
                self._project_connections[project_id].discard(websocket)

    async def broadcast_to_project(
        self, project_id: str, message: dict, exclude: WebSocket | None = None
    ):
        """
        Broadcast a message to all connections subscribed to a project.

        Args:
            project_id: The project to broadcast to
            message: The message dict to send
            exclude: Optional websocket to exclude from broadcast
        """
        if project_id not in self._project_connections:
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        # Get list of connections (snapshot to avoid lock during send)
        async with self._lock:
            connections = list(self._project_connections.get(project_id, set()))

        # Send to all connections
        disconnected = []
        for websocket in connections:
            if websocket == exclude:
                continue

            success = await self._send_to_websocket(websocket, message)
            if not success:
                disconnected.append(websocket)

        # Clean up disconnected sockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def send_to_user(self, user_id: str, message: dict):
        """
        Send a message to all connections for a specific user.

        Args:
            user_id: The user to send to
            message: The message dict to send
        """
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        async with self._lock:
            connections = [
                info.websocket for info in self._connection_info.values() if info.user_id == user_id
            ]

        disconnected = []
        for websocket in connections:
            success = await self._send_to_websocket(websocket, message)
            if not success:
                disconnected.append(websocket)

        for websocket in disconnected:
            await self.disconnect(websocket)

    async def _send_to_websocket(self, websocket: WebSocket, message: dict) -> bool:
        """
        Send a message to a specific WebSocket.

        Returns:
            True if message was sent successfully
        """
        try:
            await websocket.send_json(message)
            self._total_messages_sent += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            return False

    def get_project_subscriber_count(self, project_id: str) -> int:
        """Get the number of connections subscribed to a project."""
        return len(self._project_connections.get(project_id, set()))

    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "active_connections": len(self._connection_info),
            "active_projects": len(self._project_connections),
            "total_connections": self._total_connections,
            "total_messages_sent": self._total_messages_sent,
            "connections_by_project": {
                project_id: len(sockets)
                for project_id, sockets in self._project_connections.items()
            },
        }

    async def start_redis_listener(self):
        """Start listening to Redis pub/sub for project updates."""
        if self._redis_listener_task is not None:
            logger.warning("Redis listener already running")
            return

        self._should_listen = True
        self._redis_listener_task = asyncio.create_task(self._redis_listener_loop())
        logger.info("Redis pub/sub listener started")

    async def stop_redis_listener(self):
        """Stop the Redis listener."""
        self._should_listen = False
        if self._redis_listener_task:
            self._redis_listener_task.cancel()
            try:
                await self._redis_listener_task
            except asyncio.CancelledError:
                pass
            self._redis_listener_task = None
        logger.info("Redis pub/sub listener stopped")

    async def _redis_listener_loop(self):
        """
        Background task that listens to Redis pub/sub and broadcasts to WebSockets.

        This bridges the gap between synchronous Celery tasks and async WebSocket connections:
        1. Celery worker publishes events to Redis channels
        2. This listener receives events from Redis
        3. Events are broadcast to WebSocket clients subscribed to the project
        """
        try:
            from cache.redis_cache import get_cache

            cache = get_cache()
            if not cache.is_connected:
                logger.error("Redis not connected, cannot start listener")
                return

            pubsub = cache.get_pubsub()
            if not pubsub:
                logger.error("Could not get Redis pubsub")
                return

            # Subscribe to all project update channels using pattern
            pubsub.psubscribe("project:*:updates")
            logger.info("Subscribed to Redis pattern: project:*:updates")

            while self._should_listen:
                try:
                    # Check for messages (non-blocking with timeout)
                    message = pubsub.get_message(timeout=1.0)

                    if message and message["type"] == "pmessage":
                        channel = (
                            message["channel"].decode("utf-8")
                            if isinstance(message["channel"], bytes)
                            else message["channel"]
                        )
                        data = message["data"]

                        # Extract project_id from channel name: "project:{project_id}:updates"
                        parts = channel.split(":")
                        if len(parts) >= 2:
                            project_id = parts[1]

                            # Decode and parse message
                            try:
                                if isinstance(data, bytes):
                                    data = data.decode("utf-8")
                                event_dict = json.loads(data) if isinstance(data, str) else data

                                # Broadcast to WebSocket clients
                                await self.broadcast_to_project(project_id, event_dict)
                                logger.info(
                                    f"Redis->WS broadcast to project {project_id}: {event_dict.get('type')}"
                                )
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to decode Redis message: {e}")

                    # Small sleep to prevent busy loop
                    await asyncio.sleep(0.01)

                except Exception as e:
                    logger.error(f"Error in Redis listener loop: {e}")
                    await asyncio.sleep(1.0)  # Back off on error

            # Cleanup
            pubsub.punsubscribe()
            pubsub.close()
            logger.info("Redis listener loop ended")

        except Exception as e:
            logger.error(f"Fatal error in Redis listener: {e}", exc_info=True)


# Singleton instance
_manager_instance: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the singleton ConnectionManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ConnectionManager()
    return _manager_instance
