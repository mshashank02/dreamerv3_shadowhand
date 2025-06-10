# File: evaluate_rssm_prediction.py  
"""
Evaluate how well a Dreamer‑V3 world‑model (RSSM + Encoder + Decoder)
predicts the next observation given (s_t, a_t).

Produces:
* Mean L2 error between predicted and true next RGB frame
* Mean SSIM (structural similarity) score

Usage (from project root):

  ```bash
  python dreamerv3/evaluate_rssm_prediction.py \
      --logdir  ~/logdir/dreamer/20250607T161746 \
      --preset  shadowhand_actual              \
      --ckpt    latest                         
  ```

Requirements:
  pip install scikit-image ruamel.yaml
"""

import argparse, os, sys, glob, pathlib, importlib
from typing import Dict, Any

import numpy as np
import jax, jax.numpy as jnp
from skimage.metrics import structural_similarity as sk_ssim

# -----------------------------------------------------------------------------
# 0.  Make project package importable  (.. / dreamerv3 / this_file.py)
# -----------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   # now `import dreamerv3` works

# -----------------------------------------------------------------------------
# 1.  Helper: load config exactly like main.py
# -----------------------------------------------------------------------------
import ruamel.yaml, elements

def load_config(yaml_path: pathlib.Path, preset: str, extra: Dict[str, Any] | None = None):
    yaml_text = yaml_path.read_text()
    cfgs_dict = ruamel.yaml.YAML(typ="safe").load(yaml_text)
    cfg = elements.Config(cfgs_dict["defaults"]).update(cfgs_dict[preset])
    if extra:
        cfg = cfg.update(extra)
    return cfg

# -----------------------------------------------------------------------------
# 2.  Latest checkpoint helper
# -----------------------------------------------------------------------------

def latest_ckpt(ckpt_dir: pathlib.Path) -> pathlib.Path:
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return ckpts[-1]

# -----------------------------------------------------------------------------
# 3.  Restore agent state
# -----------------------------------------------------------------------------

def restore_agent(agent, ckpt_file: pathlib.Path):
    from embodied.core import checkpoint as ckpt_lib
    ckpt_lib.Checkpoint(ckpt_file.parent).restore(agent, ckpt_file)

# -----------------------------------------------------------------------------
# 4.  Compute L2 & SSIM between predicted and true images
# -----------------------------------------------------------------------------

def evaluate(agent, env, steps: int = 100):
    obs_list, next_list, act_list = [], [], []
    obs, _ = env.reset()
    for _ in range(steps):
        act = agent.policy(obs)
        nxt, _, term, trunc, _ = env.step(act)
        obs_list.append(obs)
        next_list.append(nxt)
        act_list.append(act)
        obs = env.reset()[0] if (term or trunc) else nxt

    # batchify (assumes 'image' key exists after wrapper)
    batch_obs  = {k: jnp.stack([o[k] for o in obs_list])  for k in obs_list[0]}
    batch_next = {k: jnp.stack([o[k] for o in next_list]) for k in next_list[0]}
    batch_act  = {k: jnp.stack([a[k] for a in act_list])  for k in act_list[0]}
    reset_mask = jnp.zeros((steps,), dtype=jnp.bool_)

    # --- encode s_t ---
    carry, _, tokens = agent.encoder({}, batch_obs, reset_mask, training=False)

    # --- one‑step observe (posterior) to get predicted next features ---
    carry, (_, feat) = agent.rssm.observe(carry, tokens, batch_act, reset_mask,
                                         training=False, single=True)

    # --- decode predicted image ---
    _, _, recon = agent.decoder({}, feat, reset_mask, training=False)
    pred_img  = np.clip(np.asarray(recon["image"]), 0.0, 1.0)  # (B,H,W,C)
    true_img  = np.asarray(batch_next["image"], dtype=np.float32) / 255.0

    l2_vals   = np.mean((pred_img - true_img) ** 2, axis=(1, 2, 3))
    ssim_vals = [sk_ssim(t, p, data_range=1.0, channel_axis=-1)
                 for p, t in zip(pred_img, true_img)]

    print("\n=== RSSM 1‑step Prediction Metrics ===")
    print(f"Mean L2   : {float(l2_vals.mean()):.5e}")
    print(f"Std  L2   : {float(l2_vals.std() ):.5e}")
    print(f"Mean SSIM : {float(np.mean(ssim_vals)):.4f}")
    print("======================================\n")

# -----------------------------------------------------------------------------
# 5.  CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="path to logdir/<timestamp>")
    parser.add_argument("--preset", required=True, help="config preset name, e.g. shadowhand_actual")
    parser.add_argument("--ckpt",   default="latest", help="checkpoint filename or 'latest'")
    parser.add_argument("--steps",  type=int, default=100, help="# real env steps to sample")
    args = parser.parse_args()

    # ---- load config identical to main.py ----
    cfg_path = PROJECT_ROOT / "dreamerv3" / "configs.yaml"
    cfg      = load_config(cfg_path, args.preset)

    # ---- build env exactly like training ----
    main_mod = importlib.import_module("dreamerv3.main")
    env      = main_mod.make_env(cfg, index=0)

    # ---- construct Agent with matching spaces ----
    from dreamerv3.agent import Agent
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    agent     = Agent(obs_space, act_space, cfg.agent)

    # ---- restore checkpoint ----
    ckpt_dir  = pathlib.Path(args.logdir) / "ckpt"
    ckpt_file = latest_ckpt(ckpt_dir) if args.ckpt == "latest" else ckpt_dir / args.ckpt
    restore_agent(agent, ckpt_file)
    print("Loaded", ckpt_file)

    # ---- run evaluation ----
    evaluate(agent, env, steps=args.steps)
