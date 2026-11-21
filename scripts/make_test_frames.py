"""Create synthetic camera frames from the reference image, for testing without a camera.

Writes:
* data/test_frames/      one frame per situation (success, wrong position, empty board, ...)
* data/demo_sequence/    the same scene in assembly order, for a dashboard playlist demo

Usage: python scripts/make_test_frames.py
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "simple_assembly.jpeg"
CX, CY, R = 746, 746, 112  # cylinder centre and radius in the reference image (pixels)
TOP_LEFT_HOLE = (322, 318)
PLAIN_BOARD = (534, 534)  # a spot of plain board between four holes


def remove_cylinder(ref: np.ndarray) -> np.ndarray:
    """Cover the cylinder with a soft-edged patch of plain board."""
    pr = R + 14
    px, py = PLAIN_BOARD
    patch = ref[py - pr : py + pr, px - pr : px + pr].astype(np.float32)
    mask = np.zeros((2 * pr, 2 * pr), np.uint8)
    cv2.circle(mask, (pr, pr), pr - 2, 255, -1)
    alpha = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32)[..., None] / 255
    out = ref.copy()
    roi = out[CY - pr : CY + pr, CX - pr : CX + pr]
    roi[:] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)
    return out


def cylinder_sprite(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = R + 4
    sprite = ref[CY - s : CY + s, CX - s : CX + s].copy()
    mask = np.zeros(sprite.shape[:2], np.uint8)
    cv2.circle(mask, (s, s), R, 255, -1)
    return sprite, mask


def paste(img: np.ndarray, sprite, mask, x: int, y: int) -> np.ndarray:
    img = img.copy()
    h, w = sprite.shape[:2]
    roi = img[y - h // 2 : y - h // 2 + h, x - w // 2 : x - w // 2 + w]
    roi[mask > 0] = sprite[mask > 0]
    return img


def scene(board: np.ndarray | None, sprite=None, mask=None, cylinder_off=False) -> np.ndarray:
    """Place the board on a grey table with distractor objects (red box, blue disc, bar)."""
    rng = np.random.default_rng(0)
    canvas = np.full((1080, 1920, 3), (150, 160, 170), np.int16)
    canvas = np.clip(canvas + rng.normal(0, 8, canvas.shape), 0, 255).astype(np.uint8)
    if board is not None:
        canvas[220:920, 610:1310] = cv2.resize(board, (700, 700))
    cv2.rectangle(canvas, (120, 300), (420, 520), (40, 40, 200), -1)
    cv2.circle(canvas, (1600, 700), 90, (200, 120, 30), -1)
    cv2.rectangle(canvas, (1450, 200), (1800, 260), (60, 60, 60), -1)
    if cylinder_off:
        small, m = cv2.resize(sprite, (110, 110)), cv2.resize(mask, (110, 110))
        roi = canvas[820:930, 1420:1530]
        roi[m > 0] = small[m > 0]
    return canvas


def main() -> None:
    ref = cv2.imread(str(REF))
    empty = remove_cylinder(ref)
    sprite, mask = cylinder_sprite(ref)
    wrong = paste(empty, sprite, mask, *TOP_LEFT_HOLE)

    test = {
        "01_success_ref.jpg": ref,
        "02_wrong_top_left.jpg": wrong,
        "03_empty_board.jpg": empty,
        "04_scene_success.jpg": scene(ref),
        "05_scene_cylinder_off_board.jpg": scene(empty, sprite, mask, cylinder_off=True),
        "06_scene_wrong_top_left.jpg": scene(wrong),
        "07_scene_no_pegboard.jpg": np.full((1080, 1920, 3), 160, np.uint8),
    }
    demo = {
        "01_no_pegboard.jpg": scene(None),
        "02_pegboard_only.jpg": scene(empty),
        "03_cylinder_off_board.jpg": test["05_scene_cylinder_off_board.jpg"],
        "04_wrong_top_left.jpg": test["06_scene_wrong_top_left.jpg"],
        "05_success.jpg": test["04_scene_success.jpg"],
    }
    for folder, frames in (("test_frames", test), ("demo_sequence", demo)):
        out = ROOT / "data" / folder
        out.mkdir(parents=True, exist_ok=True)
        for name, img in frames.items():
            cv2.imwrite(str(out / name), img)
        print(f"Wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
