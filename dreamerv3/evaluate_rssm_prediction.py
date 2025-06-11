import os
import pathlib
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from skimage.metrics import structural_similarity as ssim
import elements
from dreamerv3.agent import Agent
import embodied


def compute_l2(pred, gt):
    return ((pred - gt) ** 2).mean()


def compute_ssim(pred, gt):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    return ssim(pred, gt, channel_axis=-1, data_range=1.0)


def evaluate_rssm_prediction(args):
    # Resolve checkpoint dir using latest.txt
    ckpt_txt = elements.Path(args.logdir) / 'ckpt' / 'latest'
    with open(ckpt_txt, 'r') as f:
        ckpt_name = f.read().strip()
    ckpt_path = elements.Path(args.logdir) / 'ckpt' / ckpt_name
    print(f"Loading checkpoint from: {ckpt_path}")

    # Create environment
    env_fn = embodied.envs.from_gymnasium.FromGymnasium
    env = env_fn(args.task)
    obs_space = env.obs_space
    act_space = env.act_space

    # Create agent
    agent = Agent(obs_space, act_space, config=elements.Config())
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(ckpt_path, keys=['agent'])

    # Rollout one step
    obs = env.reset()
    carry = agent.rssm.initial(1)
    carry, _, tokens = agent.encoder(carry, obs[None], reset=np.array([[True]]), training=False, single=True)
    carry, feat, _ = agent.rssm.observe(carry, tokens, obs['action'][None], reset=np.array([[True]]), training=False, single=True)

    # Predict next step
    action = agent.actor(feat, training=False)
    carry_pred, feat_pred, _ = agent.rssm.imagine(carry, action, length=1, training=False, single=True)

    # Decode predictions and ground truth
    _, _, recon_pred = agent.decoder({}, feat_pred, reset=np.array([[True]]), training=False, single=True)
    _, _, recon_true = agent.decoder({}, feat, reset=np.array([[True]]), training=False, single=True)

    for key in recon_pred:
        pred = np.array(recon_pred[key])
        true = np.array(recon_true[key])
        if pred.ndim == 4 and pred.shape[-1] in [1, 3]:
            l2 = compute_l2(pred, true)
            ssim_val = compute_ssim(pred[0], true[0])
            print(f"[{key}] L2: {l2:.4f} | SSIM: {ssim_val:.4f}")
        else:
            print(f"[{key}] Skipping non-image key")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='latest')
    args = parser.parse_args()

    evaluate_rssm_prediction(args)
