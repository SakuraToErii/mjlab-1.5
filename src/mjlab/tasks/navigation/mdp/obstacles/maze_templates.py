from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedMazeTemplate:
  """Fixed cylinder-wall maze template in env-local coordinates."""

  name: str
  wall_centers_xy: tuple[tuple[float, float], ...]
  entrance_xy: tuple[float, float]
  entrance_yaw: float
  exit_xy: tuple[float, float]
  exit_yaw: float
  randomize_walls: bool = False


def _line_points(
  start: tuple[float, float],
  end: tuple[float, float],
  spacing: float = 1.2,
) -> list[tuple[float, float]]:
  """Create evenly spaced pillar centers along an axis-aligned wall segment."""
  x0, y0 = start
  x1, y1 = end
  length = abs(x1 - x0) + abs(y1 - y0)
  count = max(2, int(round(length / spacing)) + 1)
  points = []
  for index in range(count):
    ratio = index / (count - 1)
    x = x0 + (x1 - x0) * ratio
    y = y0 + (y1 - y0) * ratio
    points.append((round(x, 3), round(y, 3)))
  return points


def _dedupe(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
  """Preserve order while removing duplicate rounded coordinates."""
  seen: set[tuple[float, float]] = set()
  unique = []
  for point in points:
    if point not in seen:
      seen.add(point)
      unique.append(point)
  return tuple(unique)


def _walls(
  *segments: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
  points: list[tuple[float, float]] = []
  for start, end in segments:
    points.extend(_line_points(start, end))
  return _dedupe(points)


FIXED_MAZE_TEMPLATES: tuple[FixedMazeTemplate, ...] = (
  FixedMazeTemplate(
    name="straight",
    wall_centers_xy=_walls(
      ((-6.0, -1.35), (6.0, -1.35)),
      ((-6.0, 1.35), (6.0, 1.35)),
    ),
    entrance_xy=(-5.5, 0.0),
    entrance_yaw=0.0,
    exit_xy=(5.5, 0.0),
    exit_yaw=0.0,
  ),
  FixedMazeTemplate(
    name="l_turn",
    wall_centers_xy=_walls(
      ((-6.0, -1.35), (4.8, -1.35)),
      ((-6.0, 1.35), (3.45, 1.35)),
      ((3.45, 1.35), (3.45, 6.0)),
      ((6.15, -1.35), (6.15, 6.0)),
    )[:30],
    entrance_xy=(-5.5, 0.0),
    entrance_yaw=0.0,
    exit_xy=(4.8, 5.3),
    exit_yaw=1.5708,
  ),
  FixedMazeTemplate(
    name="s_curve",
    wall_centers_xy=_walls(
      ((-6.5, -5.2), (6.5, -5.2)),
      ((-6.5, -2.5), (5.0, -2.5)),
      ((-5.0, -1.35), (6.5, -1.35)),
      ((-6.5, 1.35), (6.5, 1.35)),
      ((-6.5, 2.5), (5.0, 2.5)),
      ((-6.5, 5.2), (6.5, 5.2)),
    )[:38],
    entrance_xy=(-5.8, -3.8),
    entrance_yaw=0.0,
    exit_xy=(5.8, 0.0),
    exit_yaw=0.0,
  ),
  FixedMazeTemplate(
    name="serpentine",
    wall_centers_xy=_walls(
      ((-7.0, -5.2), (7.0, -5.2)),
      ((-7.0, -2.5), (5.4, -2.5)),
      ((-5.4, -1.35), (7.0, -1.35)),
      ((-7.0, 1.35), (5.4, 1.35)),
      ((-5.4, 2.5), (7.0, 2.5)),
      ((-7.0, 5.2), (7.0, 5.2)),
    )[:48],
    entrance_xy=(-6.2, -3.85),
    entrance_yaw=0.0,
    exit_xy=(6.2, 3.85),
    exit_yaw=0.0,
  ),
  FixedMazeTemplate(
    name="dense_arena",
    wall_centers_xy=(),
    entrance_xy=(-7.0, -7.0),
    entrance_yaw=0.7854,
    exit_xy=(7.0, 7.0),
    exit_yaw=0.7854,
    randomize_walls=True,
  ),
)


def get_fixed_maze_template(level: int) -> FixedMazeTemplate:
  """Return a template, clamping out-of-range curriculum levels."""
  level = max(0, min(level, len(FIXED_MAZE_TEMPLATES) - 1))
  return FIXED_MAZE_TEMPLATES[level]
