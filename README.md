🎮 Super Mario Bros - Deep Q-Learning Agent (DQN)

Bu proje, klasik Super Mario Bros oyununu oynamayı öğrenen bir yapay zeka ajanı geliştirmeyi amaçlamaktadır. Ajan, Deep Q-Network (DQN) algoritması ile eğitilmiştir. Takviyeli öğrenme (Reinforcement Learning) temelli bu çalışma, görsel girdilerden öğrenip strateji geliştiren bir yapay zeka modelinin oyun ortamında nasıl çalıştığını göstermektedir.

📌 Özellikler

- 🎯 Takviyeli öğrenme ile oyun oynayan ajan
- 🧠 Görsel girdilerle çalışan CNN tabanlı DQN mimarisi
- 🔁 Replay memory ile stabil öğrenme süreci
- 🎯 Ödül ve ceza sistemi ile akıllı davranış geliştirme
- 🧩 Gelişmiş çevre ön işleme: Gri tonlama, çerçeve yığılma, frame skipping
- 💾 En iyi modelin otomatik kaydı
- 🧪 Test aşaması ve performans ölçümü

📂 Proje Yapısı

train.py               - Modeli eğitir
test.py                - Eğitilen modeli test eder
model.py               - DQN mimarisi (CNN + FC)
wrappers.py            - Gym ortamı için özel çerçeve işlemleri
models/                - Kayıtlı modeller (.pth dosyaları)
README.txt             - Bu açıklama dosyası

⚙️ Kullanılan Teknolojiler

- Python
- PyTorch
- OpenAI Gym
- gym-super-mario-bros
- NES-Py
- NumPy, Matplotlib

🚀 Başlangıç

Kurulum:

pip install -r requirements.txt

Eğitim:

python train.py --episodes 1000

Eğitim sırasında model her 50 bölümde bir kaydedilir ve en iyi model ayrı olarak 'models/mario_model_best.pth' dosyasına yazılır.

Test:

python test.py --model models/mario_model_best.pth --episodes 5

🧠 Model Mimarisi

Girdi: 4x84x84 boyutunda son 4 frame
3 adet Conv2D + BatchNorm katmanı
2 adet Fully Connected (FC) katman
Çıkış: 9 farklı aksiyon için Q-değeri

Model, klasik DQN yapısını kullanır ve oyun ekranındaki değişimlerden anlam çıkararak karar verir.

🎯 Ödül Sistemi

Olay                      - Ödül/Ceza
-------------------------------------
X pozisyonunda ilerleme  - +
Bölümü bitirme           - +50
Ölüm                     - -25
Takılma                  - -1
Mario büyüme / ateş alma - +5 / +10
Zaman geçmesi            - -0.005

📈 Eğitim Sonuçları

- En iyi ajan, X pozisyonuna göre en ileri gidebilen ve yüksek toplam ödül elde edendir.
- Test sırasında X-pozisyonu, skor ve toplam ödül gibi metrikler raporlanır.

📌 Notlar

- Eğitim süresi GPU’ya göre değişebilir. CUDA destekli cihaz önerilir.
- SkipFrames, PreprocessFrame ve StackFrames ile çevre ön işlemesi yapılır.
- ReplayMemory ile önceki deneyimler tekrar kullanılır.

📬 İletişim

Eğer bu projeyle ilgileniyorsanız veya katkı sağlamak isterseniz, lütfen bir issue açın ya da PR gönderin!

📄 Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
