"""Box and pegboard-grid geometry (pure Python, no model dependencies).

Boxes are (x1, y1, x2, y2) in Qwen's 0-1000 coordinates, relative to the image size.
"""

from __future__ import annotations

from thesis.knowledge.ontology import GRID

Box = tuple[float, float, float, float]


def grid_cell(cylinder: Box, pegboard: Box) -> str | None:
    """3x3 grid cell of the cylinder centre inside the pegboard box, or None if outside."""
    cx, cy = (cylinder[0] + cylinder[2]) / 2, (cylinder[1] + cylinder[3]) / 2
    x1, y1, x2, y2 = pegboard
    if x2 <= x1 or y2 <= y1:
        return None
    u, v = (cx - x1) / (x2 - x1), (cy - y1) / (y2 - y1)
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return GRID[min(int(v * 3), 2)][min(int(u * 3), 2)]


def cell_center(pegboard: Box, row: int, col: int) -> tuple[float, float]:
    """Centre of a grid cell in 0-1000 image coordinates."""
    x1, y1, x2, y2 = pegboard
    return x1 + (col + 0.5) * (x2 - x1) / 3, y1 + (row + 0.5) * (y2 - y1) / 3


def to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a 0-1000 box to pixel coordinates."""
    return (
        int(box[0] * width / 1000),
        int(box[1] * height / 1000),
        int(box[2] * width / 1000),
        int(box[3] * height / 1000),
    )
