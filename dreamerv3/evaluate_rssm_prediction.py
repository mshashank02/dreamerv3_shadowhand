# File: eval_rssm_transition.py
# Usage:
#   python eval_rssm_transition.py \
#     --logdir  ~/logdir/20250607T161746 \
#     --task    gymnasium_HandReach-v2 \
#     --ckpt    latest              # or "checkpoint_009000.npz"
#
# Requires: pip install scikit-image

import sys
import argparse, os, glob, importlib, functools
import numpy as np
import jax, jax.numpy as jnp
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------------------#
# 1.  Helpers – tiny wrappers around your dreamerv3_shadowhand codebase
# -----------------------------------------------------------------------------#
def load_cfg_and_agent(root, task):
    """Dynamically import your code and create an agent."""
    sys.path.insert(0, root)                       # make package importable
    main_mod   = importlib.import_module('dreamerv3.main')
    agent_mod  = importlib.import_module('dreamerv3.agent')
    cfg_loader = importlib.import_module('elements').Config
    configs    = cfg_loader('dreamerv3/configs.yaml')
    config     = cfg_loader(configs['defaults']).update({'task': task})
    agent      = agent_mod.Agent(
        obs_space={}, act_space={}, config=config.agent)   # minimal init
    return config, agent, main_mod.make_env

def latest_ckpt(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_*.npz')))
    assert ckpts, f'no checkpoints in {ckpt_dir}'
    return ckpts[-1]

def restore(agent, ckpt_file):
    from embodied.core import checkpoint
    checkpoint.Checkpoint(os.path.dirname(ckpt_file)).restore(agent, ckpt_file)

# -----------------------------------------------------------------------------#
# 2.  Evaluation core
# -----------------------------------------------------------------------------#
def l2(a, b):   return np.mean((a.astype(np.float32) - b.astype(np.float32))**2)

def evaluate(agent, env, steps=50):
    obs_list, nxt_list, act_list = [], [], []
    obs, _ = env.reset()
    for _ in range(steps):
        act = agent.policy(obs)                    # assumes agent.policy exists
        nxt, _, term, trunc, _ = env.step(act)
        obs_list.append(obs); nxt_list.append(nxt); act_list.append(act)
        obs = nxt if not(term or trunc) else env.reset()[0]

    # batchify
    batch_obs = {k:jnp.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    batch_nxt = {k:jnp.stack([o[k] for o in nxt_list]) for k in nxt_list[0]}
    batch_act = {k:jnp.stack([a[k] for a in act_list]) for k in act_list[0]}

    # encode current state
    _, _, token = agent.encoder({}, batch_obs, jnp.zeros((steps,), bool),
                                training=False)
    carry, (_, feat) = agent.rssm.observe(
        agent.rssm.initial(steps), token, batch_act,
        jnp.zeros((steps,), bool), training=False, single=True)

    _, _, recon = agent.decoder({}, feat, jnp.zeros((steps,), bool),
                                training=False, single=True)
    pred_img  = recon['image']          # (B,H,W,C) float in [0,1]
    true_img  = batch_nxt['image'] / 255.0

    l2s   = [l2(p,t) for p,t in zip(pred_img, true_img)]
    ssims = [ssim(t, p, channel_axis=-1, data_range=1.0)
             for p, t in zip(pred_img, true_img)]

    print(f'L2   mean {np.mean(l2s):.4e}  ±{np.std(l2s):.2e}')
    print(f'SSIM mean {np.mean(ssims):.3f}')

# -----------------------------------------------------------------------------#
# 3.  CLI
# -----------------------------------------------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--task',   required=True)
    parser.add_argument('--ckpt',   default='latest')
    args = parser.parse_args()

    root = os.path.dirname(__file__)      # project root
    config, agent, make_env = load_cfg_and_agent(root, args.task)

    # build real env (index 0, no overrides)
    env = make_env(config, index=0)

    # restore weights
    ckpt_dir  = os.path.join(args.logdir, 'ckpt')
    ckpt_file = latest_ckpt(ckpt_dir) if args.ckpt == 'latest' \
                 else os.path.join(ckpt_dir, args.ckpt)
    restore(agent, ckpt_file)
    print('Loaded checkpoint', ckpt_file)

    # run evaluation
    evaluate(agent, env, steps=100)
