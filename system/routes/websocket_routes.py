# src/yoloDetectionV3/views/websocket_routes.py
from fastapi import APIRouter, WebSocket
from system.controllers.websocket_controller import websocket_controller

router = APIRouter()

@router.websocket("/ws-realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket_controller.handle_websocket_connection(websocket)