# Log

## 2026-06-13 - Stair TrainView floor plane fix

- Symptom: local `watch_remote` / TrainView showed the stair pit covered by a transparent flat plane at the pit top, and the robot was launched upward after spawning in the inverted-pyramid stair scene.
- Root cause: the SerialLeg MJCF used by the target repo still contained its standalone world `<geom name="floor" type="plane" ...>`. MJLab already provides terrain separately, so this infinite MJCF plane overlapped the generated stair pit and acted as an unintended collision barrier.
- Fix: `src/se3_train/robot_cfg.py` now loads the MJCF through `_serialleg_spec_for_training()` and deletes the worldbody `floor` geom before creating the entity, matching the source repo behavior.
- Validation: static `ruff` check passed, and direct MJCF spec inspection confirmed `has_floor=False` for `serialleg_fourbar_surrogate_stair_visualbase_coacd_train.xml`.
