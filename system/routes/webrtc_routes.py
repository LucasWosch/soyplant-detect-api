# src/yoloDetectionV3/routes/webrtc_routes.py

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Union

import av
import numpy as np
import cv2

from fastapi import APIRouter, HTTPException, Depends
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

# IMPORTS PARA SALVAR HISTÓRICO
import system.auth as auth
from system.models.database_models import Utilizador, Analise
from system.database import get_db_session  # vamos usar o mesmo dependency, mas manualmente

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webrtc", tags=["WebRTC"])

rtc_config = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)

pcs: Dict[str, RTCPeerConnection] = {}

# pasta onde os vídeos WebRTC processados serão salvos
PROCESSED_DIR = "processed_videos"


# =======================
# Track com YOLO + SORT
# =======================
class YoloVideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        source_track: MediaStreamTrack,
        imgsz: int = 640,
        stats_channel: Optional[RTCDataChannel] = None,
    ):
        super().__init__()
        self.track = source_track
        self.imgsz = imgsz
        self.tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
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
                    asyncio.create_task(self.stats_channel.send(json.dumps(payload)))
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
    # metadados opcionais para salvar no histórico
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    local_texto: Optional[str] = None


class Answer(BaseModel):
    sdp: str
    type: str


async def salvar_analise_webrtc(
    utilizador_id: int,
    video_path: str,
    contagem_total_unicos: int,
    latitude: Optional[float],
    longitude: Optional[float],
    local_texto: Optional[str],
):
    """
    Cria um registro de Analise no banco para uma sessão WebRTC.
    Usa o mesmo get_db_session que as outras rotas, mas sem Depends.
    """
    # get_db_session é um async generator que yielda AsyncSession
    async for db in get_db_session():
        try:
            analise = Analise(
                utilizador_id=utilizador_id,
                nome_arquivo_original="webrtc_realtime",
                video_salvo_em=video_path,
                data_analise=datetime.utcnow(),
                contagem_total_unicos=contagem_total_unicos,
                latitude=latitude,
                longitude=longitude,
                local_texto=local_texto,
            )
            db.add(analise)
            await db.commit()
            await db.refresh(analise)

            logger.info(
                f"[WebRTC] Análise {analise.id} salva para utilizador {utilizador_id}. "
                f"Contagem: {contagem_total_unicos} pés únicos. Video: {video_path}"
            )
        finally:
            # sai do generator depois da primeira sessão
            break


@router.post("/offer", response_model=Answer)
async def offer_endpoint(
    offer: Offer,
    current_user: Utilizador = Depends(auth.get_utilizador_atual),
) -> Answer:
    """
    Recebe SDP Offer, cria PeerConnection com YOLO+SORT,
    grava o vídeo processado em processed_videos e,
    ao finalizar a conexão, salva histórico em Analise.
    """
    if not yolo_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo YOLO não carregado")

    pc = RTCPeerConnection(configuration=rtc_config)
    pcs[str(id(pc))] = pc
    logger.info("PC criado: %s (pcs ativos: %d)", id(pc), len(pcs))

    # garante pasta
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # nome único para o vídeo processado desta sessão
    file_name = f"webrtc_{int(time.time())}_{current_user.id}.mp4"
    output_path = os.path.join(PROCESSED_DIR, file_name)

    # DataChannel para stats
    stats_channel: RTCDataChannel = pc.createDataChannel("stats")
    logger.info("DataChannel 'stats' criado para PC %s", id(pc))

    # gravador do vídeo processado
    recorder: Union[MediaRecorder, MediaBlackhole] = MediaRecorder(output_path)

    # contexto da análise, guardado no PC para usar no cleanup
    pc._analysis_ctx = {
        "user_id": current_user.id,
        "latitude": offer.latitude,
        "longitude": offer.longitude,
        "local_texto": offer.local_texto,
        "video_path": output_path,
        "yolo_track": None,  # será preenchido em on_track
    }

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
                stats_channel=stats_channel,
            )
            # guarda referência do track no contexto para pegar os seen_ids no final
            if hasattr(pc, "_analysis_ctx") and isinstance(pc._analysis_ctx, dict):
                pc._analysis_ctx["yolo_track"] = yolo_track

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
    recorder: Optional[Union[MediaRecorder, MediaBlackhole]] = None,
) -> None:
    # para não cair duas vezes aqui para o mesmo PC
    pc_id = str(id(pc))
    ctx = getattr(pc, "_analysis_ctx", None)

    try:
        if recorder:
            await recorder.stop()
    except Exception:
        pass

    try:
        await pc.close()
    except Exception:
        pass

    pcs.pop(pc_id, None)
    logger.info("PC %s fechado (pcs ativos: %d)", pc_id, len(pcs))

    # se houver contexto de análise, salva no DB
    if isinstance(ctx, dict):
        try:
            user_id = ctx.get("user_id")
            latitude = ctx.get("latitude")
            longitude = ctx.get("longitude")
            local_texto = ctx.get("local_texto")
            video_path = ctx.get("video_path")
            yolo_track: Optional[YoloVideoTransformTrack] = ctx.get("yolo_track")

            if user_id and video_path and yolo_track is not None:
                contagem_total_unicos = len(yolo_track.seen_ids)
                await salvar_analise_webrtc(
                    utilizador_id=user_id,
                    video_path=video_path,
                    contagem_total_unicos=contagem_total_unicos,
                    latitude=latitude,
                    longitude=longitude,
                    local_texto=local_texto,
                )
            else:
                logger.info(
                    f"[WebRTC] Não foi possível salvar análise: "
                    f"user_id={user_id}, video_path={video_path}, yolo_track={yolo_track}"
                )
        except Exception as e:
            logger.exception(f"[WebRTC] Erro ao salvar análise no cleanup_pc: {e}")
