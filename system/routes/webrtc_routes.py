import asyncio
import logging
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
)
from aiortc.contrib.media import MediaBlackhole, MediaRecorder

from system.services.yolo_service import yolo_service
from system.utils.sort import Sort
from system.utils.vis import draw_tracks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

# ---- Config ICE (STUN) ----
rtc_config = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

# (opcional) guardar PCs para debug/limpeza
pcs: Dict[str, RTCPeerConnection] = {}

# ---------- Track que aplica YOLO + SORT em cada frame ----------
class YoloVideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source_track: MediaStreamTrack, imgsz: int = 640):
        super().__init__()
        self.track = source_track
        self.imgsz = imgsz
        # SORT com params pedidos
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        # FPS
        self.last_time = None
        self.fps = 0.0

    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # --- FPS suavizado ---
        import time
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * inst
        self.last_time = now

        # Redimensiona (opcional) antes da inferência para ganhar FPS
        h, w, _ = img.shape
        scale = 1.0
        if max(h, w) > self.imgsz:
            scale = self.imgsz / max(h, w)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = img

        try:
            # YOLO (Ultralytics aceita numpy BGR)
            results = yolo_service.model(img_small, verbose=False)

            # Extrai detecções no formato [x1,y1,x2,y2,score] em coordenadas da imagem original
            dets = np.empty((0, 5), dtype=np.float32)
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = boxes.conf.cpu().numpy().astype(np.float32) if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
                if scale != 1.0:
                    # reescala para o tamanho original do frame
                    xyxy /= scale
                dets = np.hstack([xyxy, conf.reshape(-1, 1)])

            # Atualiza o tracker
            tracks = self.tracker.update(dets)  # (M,5): [x1,y1,x2,y2,id]

            # Desenha as caixas rastreadas (em cima do frame original)
            draw_tracks(img, tracks, color=(255, 200, 0))

        except Exception as e:
            logger.exception("Falha no pipeline YOLO+SORT: %s", e)

        # --- Desenha FPS no canto superior esquerdo ---
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

        # Retorna para VideoFrame
        out = av.VideoFrame.from_ndarray(img, format="bgr24")
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out


# ---------- Modelos Pydantic ----------
class Offer(BaseModel):
    sdp: str
    type: str  # "offer"

class Answer(BaseModel):
    sdp: str
    type: str  # "answer"


@router.post("/offer", response_model=Answer)
async def offer_endpoint(offer: Offer) -> Answer:
    """
    Recebe um SDP Offer do navegador, cria um PeerConnection,
    pluga o transform de YOLO+SORT e devolve o SDP Answer.
    """
    if not yolo_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo YOLO não carregado")

    pc = RTCPeerConnection(configuration=rtc_config)
    pcs[str(id(pc))] = pc
    logger.info("PC criado: %s (pcs ativos: %d)", id(pc), len(pcs))

    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("PC %s - state: %s", id(pc), pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_pc(pc, recorder)

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info("Track recebida: %s (%s)", track.kind, id(track))
        if track.kind == "video":
            pc.addTrack(YoloVideoTransformTrack(track, imgsz=640))
        elif track.kind == "audio":
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            logger.info("Track finalizada: %s", track.kind)

    # SDP remoto (offer)
    remote_desc = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(remote_desc)

    # Inicia blackhole (áudio, se vier)
    await recorder.start()

    # Cria e seta answer local
    answer_obj = await pc.createAnswer()
    await pc.setLocalDescription(answer_obj)

    # Retorna SDP answer
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
