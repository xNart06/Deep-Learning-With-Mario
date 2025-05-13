import gym
import numpy as np
import cv2
from collections import deque
from gym import spaces

class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip):
        super(SkipFrames, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessFrame, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        processed = np.expand_dims(resized, axis=-1)  # (84, 84, 1)
        return processed.astype(np.uint8)


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super(StackFrames, self).__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(stack_size, 84, 84),  # PyTorch formatı: (C, H, W)
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)  # Sadece obs al
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return self.observation(None), {}  # Geriye obs ve boş info döndür

    def observation(self, obs):
        if obs is not None:
            self.frames.append(obs)
        # (stack_size, 84, 84) şekline getiriyoruz
        return np.squeeze(np.array(self.frames), axis=-1).astype(np.uint8)
