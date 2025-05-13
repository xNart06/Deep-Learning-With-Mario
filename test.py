import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import os

# Yerel modülleri içe aktar
from model import DQN
from wrappers import SkipFrames, PreprocessFrame, StackFrames

# Sabitler
CUSTOM_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['A']]
FRAME_SKIP = 4
STACK_SIZE = 4

def create_mario_env(render_mode="human"):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode=render_mode)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    env = SkipFrames(env, skip=FRAME_SKIP)
    env = PreprocessFrame(env)
    env = StackFrames(env, stack_size=STACK_SIZE)
    return env

def test_model(model_path="models/mario_model_best.pth", episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = create_mario_env()
    
    # Model oluştur ve yükle
    model = DQN(STACK_SIZE, len(CUSTOM_MOVEMENT)).to(device)
    
    # Model dosyasını kontrol et
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
        
    # Model yükle
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path} (episode {checkpoint.get('episode', 'unknown')})")
    except:
        try:
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    model.eval()  # Eval modunda
    
    all_rewards = []
    all_x_pos = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0
        prev_info = {"x_pos": 0, "score": 0, "status": "small", "life": 1}
        
        while not done:
            # Durumu torch tensöre çevir
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            
            # En iyi aksiyonu seç (epsilon-greedy olmadan)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.max(1)[1].item()
            
            # Aksiyonu uygula
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Özel ödül hesaplama (train fonksiyonundaki ile aynı)
            custom_reward = compute_reward(info, prev_info)
            prev_info = info.copy()
            
            total_reward += custom_reward
            state = next_state
            steps += 1
            
            # İlerleme göster
            if steps % 20 == 0:
                print(f"\rEpisode {episode+1}/{episodes} | Step {steps} | X-Pos: {info.get('x_pos', 0)} | Score: {info.get('score', 0)}", end="")
        
        print(f"\nEpisode {episode+1} completed with {steps} steps")
        print(f"Final score: {info.get('score', 0)} | X-Pos: {info.get('x_pos', 0)} | Total Reward: {total_reward:.2f}")
        
        all_rewards.append(total_reward)
        all_x_pos.append(info.get('x_pos', 0))
    
    env.close()
    
    # İstatistikleri yazdır
    print("\n===== Test Results =====")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"Average X position: {sum(all_x_pos)/len(all_x_pos):.2f}")
    print(f"Max X position: {max(all_x_pos)}")
    
    return all_rewards, all_x_pos

def compute_reward(info, prev_info):
    # Temel oyun ödülünü al
    reward = info["reward"] if "reward" in info else 0
    
    # X pozisyonuna göre ilerleme ödülü
    x_progress = info.get("x_pos", 0) - prev_info.get("x_pos", 0)
    reward += x_progress * 0.1
    
    # Puana göre ödül
    score_diff = info.get("score", 0) - prev_info.get("score", 0)
    reward += score_diff * 0.025
    
    # Bölümü bitirme ödülü
    if info.get("flag_get", False):
        reward += 50.0
    
    # Ölüm cezası
    if info.get("life", 1) < prev_info.get("life", 1) or info.get("status", "small") == "dead":
        reward -= 25.0
    
    # Mario büyüdüğünde ödül
    if info.get("status", "small") == "big" and prev_info.get("status", "small") == "small":
        reward += 5.0
    
    # Mario ateş gücü aldığında ödül
    if info.get("status", "small") == "fire" and prev_info.get("status", "small") != "fire":
        reward += 10.0
    
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Mario model")
    parser.add_argument("--model", type=str, default="models/mario_model_best.pth", help="Path to the model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    args = parser.parse_args()
    
    try:
        test_model(args.model, args.episodes)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()