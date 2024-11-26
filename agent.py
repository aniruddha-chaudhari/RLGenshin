import gym
from genshin_env import GenshinEnv
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage  # New imports
import time  # New import

def train_agent():
    env = GenshinEnv()
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)  # Wrap with VecTransposeImage
    model = sb3.PPO("CnnPolicy", env, verbose=1)
    try:
        model.learn(total_timesteps=100000)  # Adjust this value as needed
    except Exception as e:
        print(f"Training failed: {e}")
    time.sleep(0.1)  # Add a small delay after training
    model.save("ppo_genshin")

if __name__ == "__main__":
    train_agent()