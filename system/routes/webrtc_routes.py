import asyncio
import json
import logging
import os
import time
from typing import Dict, Optional, Union

import av
import numpy as np
import cv2

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel,
)
from aiortc.contrib.media import MediaBlackhole, MediaRecorder

from system.services.yolo_service import yolo_service
from system.utils.sort import Sort
from system.utils.vis import draw_tracks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

rtc_config = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

pcs: Dict[str, RTCPeerConnection] = {}

# =======================
# Track com YOLO + SORT
# =======================
class YoloVideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        source_track: MediaStreamTrack,
        imgsz: int = 640,
        stats_channel: Optional[RTCDataChannel] = None
    ):
        super().__init__()
        self.track = source_track
        self.imgsz = imgsz
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.last_time = None
        self.fps = 0.0

        # stats
        self.stats_channel = stats_channel
        self.seen_ids: set[int] = set()
        self.frame_index: int = 0

    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        import time as _time
        now = _time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * inst
        self.last_time = now

        h, w, _ = img.shape
        scale = 1.0
        if max(h, w) > self.imgsz:
            scale = self.imgsz / max(h, w)
            img_small = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img_small = img

        try:
            results = yolo_service.model(img_small, verbose=False)

            dets = np.empty((0, 5), dtype=np.float32)
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = (
                    boxes.conf.cpu().numpy().astype(np.float32)
                    if boxes.conf is not None
                    else np.ones((xyxy.shape[0],), dtype=np.float32)
                )
                if scale != 1.0:
                    xyxy /= scale
                dets = np.hstack([xyxy, conf.reshape(-1, 1)])

            tracks = self.tracker.update(dets)  # [x1,y1,x2,y2,id]

            # acumula IDs únicos
            if tracks is not None and len(tracks) > 0:
                for row in tracks:
                    try:
                        self.seen_ids.add(int(row[4]))
                    except Exception:
                        pass

            draw_tracks(img, tracks, color=(255, 200, 0))

            # envia stats via DataChannel (não bloqueia o pipeline)
            self.frame_index += 1
            if self.stats_channel and self.stats_channel.readyState == "open":
                payload = {
                    "type": "stats",
                    "frame_index": self.frame_index,
                    "sort_unique_ids": len(self.seen_ids),
                    "fps": float(self.fps),
                }
                try:
                    asyncio.create_task(
                        self.stats_channel.send(json.dumps(payload))
                    )
                except Exception as e:
                    logger.warning("Falha ao enviar stats no DataChannel: %s", e)

        except Exception as e:
            logger.exception("Falha no pipeline YOLO+SORT: %s", e)

        cv2.putText(
            img,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out = av.VideoFrame.from_ndarray(img, format="bgr24")
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out


class Offer(BaseModel):
    sdp: str
    type: str


class Answer(BaseModel):
    sdp: str
    type: str


@router.post("/offer", response_model=Answer)
async def offer_endpoint(offer: Offer) -> Answer:
    if not yolo_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo YOLO não carregado")

    pc = RTCPeerConnection(configuration=rtc_config)
    pcs[str(id(pc))] = pc
    logger.info("PC criado: %s (pcs ativos: %d)", id(pc), len(pcs))

    # DataChannel para stats
    stats_channel: RTCDataChannel = pc.createDataChannel("stats")
    logger.info("DataChannel 'stats' criado para PC %s", id(pc))

    recorder = MediaBlackhole()  # ou MediaRecorder se quiser gravar

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("PC %s - state: %s", id(pc), pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_pc(pc, recorder)

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info("Track recebida: %s (%s)", track.kind, id(track))
        if track.kind == "video":
            yolo_track = YoloVideoTransformTrack(
                track,
                imgsz=640,
                stats_channel=stats_channel
            )
            pc.addTrack(yolo_track)
            recorder.addTrack(yolo_track)
        elif track.kind == "audio":
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            logger.info("Track finalizada: %s", track.kind)

    remote_desc = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(remote_desc)

    await recorder.start()

    answer_obj = await pc.createAnswer()
    await pc.setLocalDescription(answer_obj)

    return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)


async def cleanup_pc(
    pc: RTCPeerConnection,
    recorder: Optional[Union[MediaRecorder, MediaBlackhole]] = None
) -> None:
    try:
        if recorder:
            await recorder.stop()
    except Exception:
        pass
    try:
        await pc.close()
    except Exception:
        pass
    pcs.pop(str(id(pc)), None)
    logger.info("PC %s fechado (pcs ativos: %d)", id(pc), len(pcs))
