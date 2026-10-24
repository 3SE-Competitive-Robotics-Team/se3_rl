"""录 recover 组 AMP 参考数据集（2026-10-24 用户定：参考 = gyi9exiz model_2200 的起身）。

用 Recover 入口（gyi9exiz 的训练环境，带域随机化）、倒地列全体当 recover 组（每局倒地出生、指令置零、8 s 截止、站稳满 1 s 结束），
不接教师，策略走学生路径（部署路径）确定性执行；每个 env 只取第一局，只留整局成功（站稳且没再倒）的，
片段截到"站稳"那一步（倾角 / 高度达标连续满 1 s，含这 1 s 站立）——成功局要跑满 10 s，后面约 8 s 都是静站，整段收会把起身稀释掉。
逐步记 19 维 AMP 运动帧（se3_train.mdp.amp_observations.amp_motion_frame，与训练时判别器看的同一个函数），
取帧口径与 se3_train.amp.AMP.process_env_step 一致：每次 env.step 之后、本步未 done 才记（出生帧随上一局 done 的那步丢掉）。
输出 se3.amp.pkl.v1（se3_train.motion_loader 读取），另附每段的出生倾角类别、地形列、env 号。
需要 GPU；在 Pod 上跑：
    python scripts/record_recover_amp_dataset.py --ckpt model_2200.pt --out assets/amp/recover_gyi9exiz_model2200/recover.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

TASK = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover"
CLASS_NAMES = ("upright", "tilted", "side", "inverted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--max-sequences", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import se3_train  # noqa: F401
    from se3_shared.amp import AMP_CONTROL_DT_S, AMP_FEATURE_NAMES, AMP_FRAME_DIM
    from se3_train.mdp.amp_observations import amp_motion_frame
    from se3_train.motion_loader import DATASET_FORMAT, MOTION_CONTRACT
    from se3_train.tasks.rough import events as ev
    from se3_train.tasks.rough import recover as rec
    from se3_train.tasks.rough.columns import column_mask

    device = "cuda:0"
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.actions["delayed_action"].teacher = None
    names = tuple(cfg.events["reset_from_fall_states"].params["terrain_type_names"])
    agent_cfg = load_rl_cfg(TASK)
    agent_cfg.logger = "tensorboard"
    raw = ManagerBasedRlEnv(cfg=cfg, device=device)
    if abs(float(raw.step_dt) - AMP_CONTROL_DT_S) > 1e-9:
        raise RuntimeError(f"控制周期 {raw.step_dt} 与 AMP 契约 {AMP_CONTROL_DT_S} 不符")
    setattr(raw, rec.RECOVER_GROUP_ATTR, column_mask(raw, names).clone())
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    n = raw.num_envs
    finished_term = raw.termination_manager
    with tempfile.TemporaryDirectory(prefix="se3-ampdata-") as tmp:
        runner = runner_cls(env, asdict(agent_cfg), tmp, device)
        runner.load(str(args.ckpt), load_cfg={"actor": True}, map_location=device)
        policy = runner.get_inference_policy(device)
        role = getattr(policy, "role_group", None)
        torch.manual_seed(args.seed)
        env.reset()
        obs = env.get_observations()
        group = getattr(raw, rec.RECOVER_GROUP_ATTR).clone()
        fall = getattr(raw, ev.FALL_SPAWN_MASK_ATTR).clone()
        if not torch.equal(group, fall & group) or int((group & ~fall).sum()) != 0:
            raise RuntimeError("recover 组里有 env 没有倒地出生")
        cls = getattr(raw, ev._FALL_CLASS_ATTR).clone()
        col = raw.scene.terrain.terrain_types.clone().long()
        episode_steps = round(
            float(cfg.terminations["recover_failed"].params["episode_s"]) / float(raw.step_dt)
        )
        frames = torch.zeros(episode_steps + 1, n, AMP_FRAME_DIM, device=device)
        length = torch.zeros(n, dtype=torch.long, device=device)
        cut = torch.zeros(n, dtype=torch.long, device=device)
        alive = group.clone()
        success = torch.zeros(n, dtype=torch.bool, device=device)
        for _k in range(episode_steps + 1):
            if not bool(alive.any()):
                break
            if role is not None and role in obs.keys():  # noqa: SIM118
                obs[role][:] = 0.0
            with torch.no_grad():
                actions = policy(obs)
            obs, _r, dones, _e = env.step(actions)
            done = dones.bool().reshape(-1)
            finished = finished_term.get_term("recover_finished").bool()
            success |= alive & done & finished
            write = alive & ~done
            frame = amp_motion_frame(raw, AMP_FEATURE_NAMES)
            frames[length.clamp(max=episode_steps)[write], write.nonzero().flatten()] = frame[write]
            length += write.long()
            just_stable = rec.recover_state(raw).just_stable
            cut = torch.where(alive & just_stable & (cut == 0), length, cut)
            alive &= ~done
    raw.close()

    ok = torch.nonzero(success & (cut >= 2)).flatten()[: args.max_sequences].tolist()
    sequences = [frames[: int(cut[i]), i].cpu().numpy().astype(np.float32) for i in ok]
    lengths = [int(s.shape[0]) for s in sequences]
    terrain_names = list(raw.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
    annotations = [
        {"env": i, "fall_class": CLASS_NAMES[int(cls[i])], "terrain": terrain_names[int(col[i])]}
        for i in ok
    ]
    payload = {
        "format": DATASET_FORMAT,
        "motion_contract": MOTION_CONTRACT,
        "fps": 1.0 / AMP_CONTROL_DT_S,
        "dt": AMP_CONTROL_DT_S,
        "frame_dim": AMP_FRAME_DIM,
        "transition_dim": 2 * AMP_FRAME_DIM,
        "feature_names": list(AMP_FEATURE_NAMES),
        "sequences": sequences,
        "lengths": lengths,
        "annotations": annotations,
        "retargeted_to_serialleg": True,
        "source": {
            "task": TASK,
            "checkpoint": str(args.ckpt),
            "seed": args.seed,
            "deterministic": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_group = int(group.sum())
    by_class = {c: sum(a["fall_class"] == c for a in annotations) for c in CLASS_NAMES}
    by_terrain = {
        t: sum(a["terrain"] == t for a in annotations)
        for t in sorted({a["terrain"] for a in annotations})
    }
    lens = np.array(lengths) * AMP_CONTROL_DT_S
    print(
        f"[amp-data] recover 组 {n_group} env，成功 {int(success.sum())}（{float(success.sum()) / max(n_group, 1):.3f}），"
        f"写出 {len(sequences)} 段 / {int(np.sum(lengths))} 帧 → {args.out}",
        flush=True,
    )
    print(f"[amp-data] 类别 {by_class} 地形 {by_terrain}", flush=True)
    print(
        f"[amp-data] 段长 s p10/p50/p90 {np.percentile(lens, [10, 50, 90]).round(2).tolist()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
