import os
import sys
import pathlib
import argparse
import numpy as np
import ruamel.yaml as yaml
import elements
import embodied

from embodied import Agent
from embodied.envs.from_gymnasium import FromGymnasium

from skimage.metrics import structural_similarity as ssim


# === CONFIG LOADER ===
def load_cfg(logdir_path, task):
    logdir = elements.Path(logdir_path)
    config_path = logdir / 'config.yaml'

    yaml_loader = yaml.YAML(typ='safe')
    raw_cfg = yaml_loader.load(config_path.read())

    config = elements.Config(raw_cfg)
    config = config.update(task=task, logdir=str(logdir))
    return config


# === MAIN EVALUATION SCRIPT ===
def evaluate_rssm_prediction(args):
    config = load_cfg(args.logdir, args.task)

    # Load environment and agent
    env = FromGymnasium(args.task)
    obs_space = env.obs_space
    act_space = env.act_space

    agent = Agent(obs_space, act_space, config.agent)
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.ckpt or args.logdir, keys=['agent'])

    # Initialize
    obs = env.reset()
    state = agent.init_policy(batch_size=1)

    # Collect N transitions from the real environment
    num_steps = args.steps
    real_obs, pred_obs = [], []
    total_l2, total_ssim = 0.0, 0.0

    for _ in range(num_steps):
        act = agent.policy(obs, state, mode='eval')
        next_obs, _, done, _ = env.step(act)

        # Predict next obs from RSSM using current obs and action
        _, _, embed = agent.encoder(state, obs, reset=False, training=False, single=True)
        _, feat = agent.rssm.observe(state, embed, act, reset=False, training=False, single=True)
        _, _, recon = agent.decoder(state, feat, reset=False, training=False, single=True)

        obs_image = obs['image'].astype(np.float32) / 255.0
        recon_image = recon['image'].mean().numpy().squeeze()

        l2_loss = np.mean((obs_image - recon_image) ** 2)
        ssim_score = ssim(obs_image, recon_image, multichannel=True)

        total_l2 += l2_loss
        total_ssim += ssim_score

        real_obs.append(obs_image)
        pred_obs.append(recon_image)

        obs = next_obs if not done else env.reset()

    print(f"Average L2 Loss: {total_l2 / num_steps:.4f}")
    print(f"Average SSIM:   {total_ssim / num_steps:.4f}")


# === ENTRY POINT ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, help='Path to logdir with config and checkpoint')
    parser.add_argument('--task', required=True, help='Gymnasium task name')
    parser.add_argument('--ckpt', default='latest', help='Checkpoint path or "latest"')
    parser.add_argument('--steps', type=int, default=100, help='Number of evaluation steps')
    args = parser.parse_args()

    evaluate_rssm_prediction(args)
