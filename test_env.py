from genshin_env import GenshinEnv
import time
from stable_baselines3 import PPO

if __name__ == "__main__":
    try:
        # Load the trained agent
        model = PPO.load("ppo_genshin")
        
        # Initialize the environment
        env = GenshinEnv()
        
        # Reset the environment to start a new episode
        obs = env.reset()
        
        # Run a few steps to test interaction
        for step in range(10):
            action, _states = model.predict(obs)
            print(f"Executing Action: {action}")  # Added debug print
            obs, reward, done, info = env.step(action)
            
            print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
            time.sleep(0.5)  # Add delay between actions
            
            if done:
                print("Episode finished!")
                break
                
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
