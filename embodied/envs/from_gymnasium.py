# from_gym.py adapted to work with Gymnasium. Differences:
#
# - gym.* -> gymnasium.*
# - Deals with .step() returning a tuple of (obs, reward, terminated, truncated,
#   info) rather than (obs, reward, done, info).
# - Also deals with .reset() returning a tuple of (obs, info) rather than just
#   obs.
# - Passes render_mode='rgb_array' to gymnasium.make() rather than .render().
# - A bunch of minor/irrelevant type checking changes that stopped pyright from
#   complaining (these have no functional purpose, I'm just a completionist who
#   doesn't like red squiggles).

import functools
from typing import Any, Generic, TypeVar, Union, cast, Dict
from PIL import Image 
import embodied
import gymnasium 
import gymnasium_robotics
import numpy as np
import elements
U = TypeVar('U')
V = TypeVar('V')


class FromGymnasium(embodied.Env, Generic[U, V]):
  def __init__(self, env: Union[str, gymnasium.Env[U, V]], obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env: gymnasium.Env[U, V] = gymnasium.make(env, render_mode="rgb_array", **kwargs)
    else:
      assert not kwargs, kwargs
      assert env.render_mode == "rgb_array", f"render_mode must be rgb_array, got {self._env.render_mode}"
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self._image_shape = None


  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      obs_space = cast(gymnasium.spaces.Dict, self._env.observation_space)
      spaces = obs_space.spaces

      # Special case: Gymnasium-Robotics style GoalEnv
      if all(key in spaces for key in ['observation', 'desired_goal', 'achieved_goal']):
        # Flatten each subspace inside the goal dict
        flat_spaces = {}
        for key, subspace in spaces.items():
          if isinstance(subspace, gymnasium.spaces.Dict):
            flat_spaces.update({f"{key}/{k}": v for k, v in subspace.spaces.items()})
          else:
            flat_spaces[key] = subspace
        spaces = flat_spaces
    else:
      spaces = {self._obs_key: self._env.observation_space}

    # Convert each Gym space to an Elements space
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    
        # ✅ Delay render until after env.reset()
    if self._image_shape is None:
        try:
            self._env.reset()
            original_shape = self._env.render().shape
            print(f"[FromGymnasium] Original image shape: {original_shape}")
            self._image_shape = (64, 64, 3)  # ✅ Set resized shape manually
        except Exception as e:
            print(f"[Warning] Could not determine image shape: {e}")
            self._image_shape = (64, 64, 3)  # fallback
            print(f"[FromGymnasium] Using fallback image shape: {self._image_shape}")

    # ✅ Now add image space
    #spaces['image'] = elements.Space(np.uint8, self._image_shape, 0, 255)

    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      act_space = cast(gymnasium.spaces.Dict, self._env.action_space)
      spaces = act_space.spaces
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      # we don't bother setting ._info here because it gets set below, once we
      # take the next .step()
      obs, _ = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      gymnasium_action = cast(V, self._unflatten(action))
    else:
      gymnasium_action = cast(V, action[self._act_key])
    obs, reward, terminated, truncated, self._info = self._env.step(gymnasium_action)
    self._done = terminated or truncated
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    
    if not self._obs_dict:
      obs = {self._obs_key: obs}

    # Special handling for GoalEnv-style observations
    elif all(k in obs for k in ['observation', 'desired_goal', 'achieved_goal']):
      # Flatten the goal observation structure
      goal_obs = {}
      for key in ['observation', 'desired_goal', 'achieved_goal']:
        value = obs[key]
        if isinstance(value, dict):
          # Nested dict inside one of the goal components
          for subkey, subval in value.items():
            goal_obs[f"{key}/{subkey}"] = subval
        else:
          goal_obs[key] = value
      obs = goal_obs

    else:
      # If not GoalEnv style, just flatten any nested dict
      obs = self._flatten(obs)

    # Convert to NumPy arrays and add meta info
    np_obs: Dict[str, Any] = {k: np.asarray(v) for k, v in obs.items()}

    try:
        image = self._env.render()
        if image is not None:
            resized = np.array(Image.fromarray(image).resize((64, 64)))
            self._latest_render = resized  # store it for use elsewhere (e.g., via .info)
            if is_first:
             print(f"[DEBUG] Rendered image shape: {np_obs['image'].shape}")
        else:
            #np_obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)
            print(f"[DEBUG] Using fallback image shape: {np_obs['image'].shape}")
    except Exception as e:
        print(f"[Warning] Render failed: {e}")
        #np_obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)
    np_obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    self._info['render'] = self._latest_render  # store it for use in logfn()

    return np_obs


  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gymnasium.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)