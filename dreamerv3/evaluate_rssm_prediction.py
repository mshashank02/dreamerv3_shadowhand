import os
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim

import embodied
import jax
import jax.numpy as jnp

from dreamerv3 import agent as agent_mod
from embodied import load_config


def load_agent(logdir, task, ckpt):
    config = embodied.Config()
    config = config.update(logdir=logdir, run=dict(task=task))
    config = config.update(task=task)
    config = config.update({'jax': {}})  # Avoid KeyError: 'jax'

    # Load config from checkpoint directory if exists
    config_path = os.path.join(logdir, 'config.yaml')
    if os.path.exists(config_path):
        config = config.update(embodied.Config(yaml=embodied.Path(config_path).read()))

    env = embodied.envs.from_gymnasium.FromGymnasium(task)
    obs_space = env.obs_space
    act_space = env.act_space
    env.close()

    agent = agent_mod.Agent(obs_space, act_space, config)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(logdir, ckpt)

    return agent, obs_space


def evaluate(agent, obs_space, logdir, batch_size=4):
    replay_dir = os.path.join(logdir, 'replay')
    replay = embodied.replay.Replay(
        length=50, capacity=10000, online=False, chunksize=50,
        directory=replay_dir)

    ssim_scores, l2_errors = [], []
    print("Collecting samples...")
    for _ in range(10):
        batch = replay.sample(batch_size, 'train')

        obs = batch['obs']  # [B, T, *]
        actions = batch['action']
        resets = batch['reset']

        carry = agent.rssm.initial(batch_size)
        carry, entries, _ = agent.rssm.observe(
            carry, batch['embed'], actions, resets, training=False)

        # Predict next observation from RSSM state
        features = {k: entries[k][:, :-1] for k in ('deter', 'stoch')}
        feats = {k: jnp.reshape(v, (-1, *v.shape[2:])) for k, v in features.items()}
        carry = {k: v[:, -1] for k, v in features.items()}

        # Decode from features
        _, _, decoded = agent.decoder({}, feats, reset=np.zeros_like(resets[:, 1:]), training=False)
        true_obs = obs['image'][:, 1:]  # Ground truth next obs

        decoded_img = np.clip(decoded['image'].mean(), 0, 1) * 255
        decoded_img = decoded_img.astype(np.uint8)

        true_img = true_obs.reshape((-1,) + true_obs.shape[2:])

        for i in range(true_img.shape[0]):
            gt = true_img[i]
            pred = decoded_img[i]
            ssim_score = ssim(gt, pred, channel_axis=-1)
            l2_error = np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2)
            ssim_scores.append(ssim_score)
            l2_errors.append(l2_error)

    print("Avg SSIM:", np.mean(ssim_scores))
    print("Avg L2 Error:", np.mean(l2_errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--ckpt', default='latest')
    args = parser.parse_args()

    agent, obs_space = load_agent(args.logdir, args.task, args.ckpt)
    evaluate(agent, obs_space, args.logdir)
