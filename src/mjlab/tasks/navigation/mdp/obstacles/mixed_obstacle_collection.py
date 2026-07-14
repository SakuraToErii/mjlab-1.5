"""Native MuJoCo fixed-base mocap obstacle entities."""

from __future__ import annotations

from functools import partial

import mujoco

from mjlab.entity import EntityCfg

# Slot counts (total = 120)
V5_NUM_CYL_SMALL = 30
V5_NUM_CYL_MEDIUM = 30
V5_NUM_CYL_LARGE = 20
V5_NUM_BOX_LOW = 20
V5_NUM_BOX_TALL = 20
V5_MAX_MIXED_OBSTACLES = 120
V5_CYL_RADIUS_SMALL = 0.25
V5_CYL_RADIUS_MEDIUM = 0.40
V5_CYL_RADIUS_LARGE = 0.55
V5_CYLINDER_HEIGHT = 2.0
V5_BOX_LOW_SIZE = (0.8, 0.8, 0.6)
V5_BOX_TALL_SIZE = (0.6, 0.6, 2.0)


def _primitive_spec(
  kind: str, size: tuple[float, float, float], color: tuple[float, float, float, float]
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="obstacle")
  geom = body.add_geom(name="collision")
  geom.type = (
    mujoco.mjtGeom.mjGEOM_CYLINDER if kind == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX
  )
  geom.size[:] = size
  geom.rgba[:] = color
  geom.group = 1
  geom.contype = 1
  geom.conaffinity = 1
  geom.friction[:] = (1.0, 0.005, 0.0001)
  return spec


def make_obstacle_entity(
  kind: str, size: tuple[float, float, float], color: tuple[float, float, float, float]
) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    spec_fn=partial(_primitive_spec, kind, size, color),
  )


def make_cylinder_obstacle_entities(
  max_obstacles: int, radius: float, height: float
) -> dict[str, EntityCfg]:
  return {
    f"cylinder_obstacle_{idx:03d}": make_obstacle_entity(
      "cylinder", (radius, height * 0.5, 0.0), (0.2, 0.2, 0.85, 1.0)
    )
    for idx in range(max_obstacles)
  }


def make_mixed_obstacle_collection() -> dict[str, EntityCfg]:
  specs: list[
    tuple[str, tuple[float, float, float], tuple[float, float, float, float]]
  ] = []
  specs += [
    (
      "cylinder",
      (V5_CYL_RADIUS_SMALL, V5_CYLINDER_HEIGHT * 0.5, 0.0),
      (0.2, 0.45, 0.85, 1.0),
    )
  ] * V5_NUM_CYL_SMALL
  specs += [
    (
      "cylinder",
      (V5_CYL_RADIUS_MEDIUM, V5_CYLINDER_HEIGHT * 0.5, 0.0),
      (0.15, 0.35, 0.9, 1.0),
    )
  ] * V5_NUM_CYL_MEDIUM
  specs += [
    (
      "cylinder",
      (V5_CYL_RADIUS_LARGE, V5_CYLINDER_HEIGHT * 0.5, 0.0),
      (0.1, 0.25, 0.95, 1.0),
    )
  ] * V5_NUM_CYL_LARGE
  low_half_size = (
    V5_BOX_LOW_SIZE[0] * 0.5,
    V5_BOX_LOW_SIZE[1] * 0.5,
    V5_BOX_LOW_SIZE[2] * 0.5,
  )
  tall_half_size = (
    V5_BOX_TALL_SIZE[0] * 0.5,
    V5_BOX_TALL_SIZE[1] * 0.5,
    V5_BOX_TALL_SIZE[2] * 0.5,
  )
  specs.extend([("box", low_half_size, (0.55, 0.35, 0.2, 1.0))] * V5_NUM_BOX_LOW)
  specs.extend([("box", tall_half_size, (0.7, 0.3, 0.15, 1.0))] * V5_NUM_BOX_TALL)
  return {
    f"mixed_obstacle_{idx:03d}": make_obstacle_entity(kind, size, color)
    for idx, (kind, size, color) in enumerate(specs)
  }
