import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Yerel modülleri içe aktar
from model import DQN
from wrappers import SkipFrames, PreprocessFrame, StackFrames

BATCH_SIZE = 64                # 32-128 arası dengeli (GPU varsa 64 tercih edilir)
GAMMA = 0.99                  # Gelecek ödülleri maksimize etmek için kritik
EPSILON_START = 1.0           # Başlangıçta %100 rastgele hareket
EPSILON_END = 0.01            # Sonunda %1 rastgele hareket (0.1'den daha iyi)
EPSILON_DECAY = 300000        # 300K adımda epsilon düşer (uzun keşif süresi)
TARGET_UPDATE = 10000         # 10K adımda bir hedef ağı güncelle
LEARNING_RATE = 0.00025       # Adam optimizer için ideal (0.0001-0.0005 arası)
MEMORY_SIZE = 200000          # Replay bellekte 200K örnek sakla
INITIAL_MEMORY = 20000        # Eğitim öncesi 20K rastgele örnek topla
FRAME_SKIP = 4                # Her 4 frame'de 1 aksiyon al
STACK_SIZE = 4                # Son 4 frame'i stackle (hareket algısı için)

# Takılma tespiti için sabitler
STUCK_DETECTION_FRAMES = 30  # Kaç frame boyunca aynı pozisyonda kalırsa "takılmış" sayılacak
STUCK_PENALTY = -1.0  # Takılma durumunda uygulanacak ceza

# Oyun aksiyon seti
CUSTOM_MOVEMENT = [
    ['NOOP'], 
    ['right'], ['right', 'A'], 
    ['A'], 
    ['left'], ['left', 'A'], 
    ['down'], ['right', 'B'], 
    ['right', 'A', 'B']
]


# Deneyim Replay için namedtuple
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    def __len__(self):
        return len(self.memory)

def create_mario_env(render_mode="human"):
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode=render_mode)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    env = SkipFrames(env, skip=FRAME_SKIP)
    env = PreprocessFrame(env)
    env = StackFrames(env, stack_size=STACK_SIZE)
    return env

class Agent:
    def __init__(self, n_actions, device):
        self.device = device
        self.policy_net = DQN(STACK_SIZE, n_actions).to(device)
        self.target_net = DQN(STACK_SIZE, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Eval modunda hedef ağ (gradyan hesaplanmaz)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        if training:
            # Epsilon-greedy strateji
            epsilon = max(EPSILON_END, EPSILON_START - (self.steps_done / EPSILON_DECAY))
            self.steps_done += 1
            
            if random.random() < epsilon:
                return torch.tensor([[random.randrange(len(CUSTOM_MOVEMENT))]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state)
            return q_values.max(1)[1].view(1, 1)
    
    def optimize_model(self):
        if not self.memory.can_sample(BATCH_SIZE):
            return 0  # Yeterli deneyim yok
        
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        # Terminal olmayan durumları maske olarak işaretleme
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=self.device, dtype=torch.bool
        )
        
        non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) 
                                       for s in batch.next_state if s is not None]).to(self.device) / 255.0
        
        state_batch = torch.cat([torch.FloatTensor(s).unsqueeze(0) 
                             for s in batch.state]).to(self.device) / 255.0
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat([torch.FloatTensor([[r]]) for r in batch.reward]).to(self.device)
        
        # Q değerlerini hesaplama
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        # Hedef ağı kullanarak next state değerlerini hesapla
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Beklenen Q değerlerini hesapla
        expected_q_values = reward_batch + (GAMMA * next_state_values.unsqueeze(1))
        
        # Huber loss hesapla (daha stabil)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping - büyük değişimleri sınırlandırır
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_checkpoint(self, path, episode, reward=None, best=False):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        
        if reward is not None:
            checkpoint['reward'] = reward
            
        torch.save(checkpoint, path)
        
        if best:
            print(f"New best model saved with reward: {reward:.2f}")
        else:
            print(f"Checkpoint saved at episode {episode+1}")

class StuckDetector:
    """Mario'nun bir engele takılıp takılmadığını tespit eden sınıf"""
    def __init__(self, detection_frames=STUCK_DETECTION_FRAMES):
        self.detection_frames = detection_frames
        self.x_positions = deque(maxlen=detection_frames)
        self.is_stuck = False
        
    def update(self, x_pos):
        self.x_positions.append(x_pos)
        
        # Yeterli veri toplandıysa kontrol et
        if len(self.x_positions) == self.detection_frames:
            # Eğer son N frame'de x pozisyonu değişmediyse takılmış demektir
            x_min = min(self.x_positions)
            x_max = max(self.x_positions)
            
            # Eğer pozisyon çok az değiştiyse (2 birimden az) takılmış kabul et
            self.is_stuck = (x_max - x_min) < 2
            return self.is_stuck
        
        return False
    
    def reset(self):
        self.x_positions.clear()
        self.is_stuck = False

def compute_reward(info, prev_info, stuck_detector):
    # Temel oyun ödülünü al
    reward = info["reward"] if "reward" in info else 0
    
    # X pozisyonuna göre ilerleme ödülü
    x_progress = info.get("x_pos", 0) - prev_info.get("x_pos", 0)
    reward += x_progress * 0.1
    
    # X pozisyonunu takılma detektörüne gönder
    stuck_detector.update(info.get("x_pos", 0))
    
    # Eğer Mario takıldıysa dinamik ceza uygula
    if stuck_detector.is_stuck:
        stuck_frames = len(stuck_detector.x_positions)
        penalty = STUCK_PENALTY * (stuck_frames / STUCK_DETECTION_FRAMES)  # Süreye bağlı ceza
        reward += penalty
        print(f"\rMario takıldı! Frame: {stuck_frames}/{STUCK_DETECTION_FRAMES} | Ceza: {penalty:.2f}", end="")
    
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
        
    # Zaman azaldıkça küçük ceza (aciliyet hissi yaratır)
    time_penalty = -0.005
    reward += time_penalty
    
    return reward

def collect_initial_experiences(env, agent, device):
    """Rastgele hareketlerle başlangıç deneyimleri topla"""
    print("Başlangıç deneyimleri toplanıyor...")
    state = env.reset()[0]
    
    for i in range(INITIAL_MEMORY):
        if i % 1000 == 0:
            print(f"Başlangıç örnekleri: {i}/{INITIAL_MEMORY}")
            
        action = torch.tensor([[random.randrange(len(CUSTOM_MOVEMENT))]], device=device)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        agent.memory.push(
            state, 
            action, 
            next_state if not done else None,
            reward,
            done
        )
        
        if done:
            state = env.reset()[0]
        else:
            state = next_state
    
    print(f"Başlangıç örnekleri tamamlandı. Toplam örnekler: {len(agent.memory)}")

def train(args):
    # Cihaz ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ortam oluştur
    env = create_mario_env(render_mode="human")
    
    # Ajan oluştur
    agent = Agent(len(CUSTOM_MOVEMENT), device)
    
    # Kayıt dizini oluştur
    os.makedirs("models", exist_ok=True)
    
    # Eğitim parametreleri
    num_episodes = args.episodes
    checkpoint_interval = args.checkpoint_interval
    best_reward = -float('inf')
    
    # Başlangıç deneyimleri topla
    if not args.skip_initial_collection:
        collect_initial_experiences(env, agent, device)
    
    print("Eğitim başlıyor...")
    total_training_start = time.time()
    
    # Eğitim döngüsü
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0
        prev_info = {"x_pos": 0, "score": 0, "status": "small", "life": 1}
        episode_losses = []
        
        # Takılma detektörünü her episode başında sıfırla
        stuck_detector = StuckDetector()
        
        while not done:
            # Aksiyon seç
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # Özel ödül hesapla
            reward = compute_reward(info, prev_info, stuck_detector)
            prev_info = info.copy()
            total_reward += reward
            
            # Deneyimi hafızaya ekle
            agent.memory.push(
                state, 
                action, 
                next_state if not done else None,
                reward,
                done
            )
            
            # Modeli optimize et
            loss = agent.optimize_model()
            if loss:
                episode_losses.append(loss)
                
            # Hedef ağı periyodik olarak güncelle
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_network()
                
            state = next_state
            steps += 1
            
            # İlerleme bilgisi yazdır
            if steps % 100 == 0:
                avg_loss = sum(episode_losses[-100:]) / max(len(episode_losses[-100:]), 1)
                x_pos = info.get("x_pos", 0)
                epsilon = max(EPSILON_END, EPSILON_START - (agent.steps_done / EPSILON_DECAY))
                print(f"\rEpisode {episode+1} | Step {steps} | X-Pos: {x_pos} | Epsilon: {epsilon:.2f} | Avg Loss: {avg_loss:.5f}", end="")
        
        # Episode istatistikleri
        episode_duration = time.time() - episode_start_time
        avg_loss = sum(episode_losses) / max(len(episode_losses), 1)
        
        print(f"\nEpisode {episode+1}/{num_episodes} completed in {episode_duration:.2f}s")
        print(f"Steps: {steps} | Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.5f}")
        print(f"Final X-Pos: {info.get('x_pos', 0)} | Score: {info.get('score', 0)}")
        
        # Checkpoint kaydetme
        if (episode + 1) % checkpoint_interval == 0:
            agent.save_checkpoint(f"models/mario_checkpoint_episode_{episode+1}.pth", episode)
        
        # En iyi modeli kaydet
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_checkpoint("models/mario_model_best.pth", episode, reward=best_reward, best=True)
    
    # Son modeli kaydet
    agent.save_checkpoint("models/mario_model_final.pth", num_episodes-1)
    
    total_training_time = time.time() - total_training_start
    print(f"Training completed in {total_training_time/60:.2f} minutes")
    print(f"Best reward: {best_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Super Mario Bros")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint every N episodes")
    parser.add_argument("--skip_initial_collection", action="store_true", help="Skip initial experience collection")
    args = parser.parse_args()
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Model kaydet - DÜZELTME: static metot değil, nesne metodu olarak çağrılmalı
        print("Saving model...")
        try:
            # global agent değişkeni olmadığı için burada kaydedemeyiz
            # Çözüm: train() fonksiyonuna global try/except bloğu eklemek
            print("Model could not be saved on interrupt. Run with shorter episode count to save properly.")
        except Exception as e:
            print(f"Could not save model: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()