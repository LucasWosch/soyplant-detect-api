import cv2
import numpy as np
from typing import Tuple

def draw_tracks(
    img: np.ndarray,
    tracks: np.ndarray,              # shape (M,5) -> [x1,y1,x2,y2,id]
    color: Tuple[int,int,int]=(255, 200, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    """
    Desenha caixas e IDs de tracking diretamente em `img` (in-place).
    """
    if tracks is None or len(tracks) == 0:
        return

    for t in tracks:
        x1, y1, x2, y2, tid = t.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f'ID {int(tid)}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        # fundo do texto
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        # texto
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
