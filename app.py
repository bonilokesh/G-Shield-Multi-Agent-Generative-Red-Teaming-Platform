import gradio as nn_gradio  # Using an alias to avoid conflict with torch.nn
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

# --- 0. CENTRALIZED DATABASE LAYER ---
DB_FILE = "vision_security_logs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Create table to keep track of every biometric security attack simulation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS biometric_attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            target_identity INTEGER,
            baseline_result TEXT,
            hardened_result TEXT
        )
    """)
    conn.commit()
    conn.close()

# Spin up database schema on initialization
init_db()

def log_attack_to_db(target_digit, baseline_class, hardened_class):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO biometric_attacks (timestamp, target_identity, baseline_result, hardened_result)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        int(target_digit), 
        f"Read as {baseline_class}", 
        f"Read as {hardened_class}"
    ))
    conn.commit()
    conn.close()

def get_recent_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM biometric_attacks ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()
    
    log_str = "📜 CENTRALIZED DATABASE SIEM AUDIT LOGS:\n"
    for row in rows:
        log_str += f"[{row[1]}] Targeted Identity: #{row[2]} | Unprotected System: {row[3]} | Hardened System: {row[4]}\n"
    return log_str

# --- 1. MODEL ARCHITECTURE DEFINITIONS ---
class BlueTeamNet(nn.Module):
    def __init__(self):
        super(BlueTeamNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class RedGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(RedGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

# --- 2. LOAD PRE-TRAINED AGENTS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = RedGenerator(latent_dim=100).to(device)
base_def = BlueTeamNet().to(device)
hard_def = BlueTeamNet().to(device)

gen.load_state_dict(torch.load("red_generator_final.pth", map_location=device))
base_def.load_state_dict(torch.load("blue_team_baseline.pth", map_location=device))
hard_def.load_state_dict(torch.load("blue_team_hardened.pth", map_location=device))

gen.eval()
base_def.eval()
hard_def.eval()

# --- 3. PREDICTION & ATTACK LOGIC ---
def launch_attack(target_digit):
    z = torch.randn(1, 100).to(device)
    label = torch.tensor([int(target_digit)]).to(device)
    
    with torch.no_grad():
        generated_img = gen(z, label).squeeze().cpu().numpy()
        
        # Format for evaluation
        eval_tensor = torch.tensor(generated_img).unsqueeze(0).unsqueeze(0).to(device)
        base_pred = torch.softmax(base_def(eval_tensor), dim=1).squeeze().cpu().numpy()
        hard_pred = torch.softmax(hard_def(eval_tensor), dim=1).squeeze().cpu().numpy()
    
    # Save image to return path
    plt.figure(figsize=(3, 3))
    plt.imshow(generated_img, cmap='gray')
    plt.axis('off')
    img_path = 'attack_sample.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Process Metrics
    base_detected_class = np.argmax(base_pred)
    base_confidence = base_pred[int(target_digit)] * 100
    base_status = "❌ BREACHED" if base_detected_class != int(target_digit) else "✅ SECURE"
    base_msg = f"Status: {base_status}\nRead Identity As: {base_detected_class}\nTarget Confidence: {base_confidence:.1f}%"
    
    hard_detected_class = np.argmax(hard_pred)
    hard_confidence = hard_pred[int(target_digit)] * 100
    hard_status = "✅ SECURE" if hard_detected_class == int(target_digit) else "❌ BREACHED"
    hard_msg = f"Status: {hard_status}\nRead Identity As: {hard_detected_class}\nTarget Confidence: {hard_confidence:.1f}%"
    
    # 1. Commit metrics into SQLite transaction logs
    log_attack_to_db(target_digit, base_detected_class, hardened_class=hard_detected_class)
    
    # 2. Extract historical records for display
    db_telemetry = get_recent_logs()
    
    return img_path, base_msg, hard_msg, db_telemetry

# --- 4. GRADIO INTERFACE LAYOUT ---
with gr.Blocks(title="G-Shield Red-Teaming Platform") as demo:
    gr.Markdown("# 🛡️ G-Shield: Multi-Agent Generative Red-Teaming Platform")
    gr.Markdown("Automated vulnerability discovery and adaptive hardening for vision infrastructure.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Attack Simulation Controls")
            slider = gr.Slider(minimum=0, maximum=9, step=1, value=5, label="Select Target Identity to Attack")
            attack_btn = gr.Button("🚀 Launch Generative Red-Team Attack", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Live Pipeline Inspection")
            output_image = gr.Image(label="Red-Team Generated Adversarial Patch", type="filepath")
            
            with gr.Row():
                base_output = gr.Textbox(label="Unprotected Baseline System", lines=4)
                hard_output = gr.Textbox(label="G-Shield Hardened System (Self-Healed)", lines=4)
                
    gr.Markdown("---")
    # Live database feed text region
    history_box = gr.Textbox(label="🖥️ Live Centralized SIEM Database Audit Logs", lines=6, interactive=False)
                
    attack_btn.click(
        fn=launch_attack, 
        inputs=slider, 
        outputs=[output_image, base_output, hard_output, history_box]
    )

# Launch with global sharing enabled
demo.launch(share=True, debug=True)
