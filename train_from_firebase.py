# train_from_firebase.py
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os

# Import your model architecture and feature parser
from model import PVModel
from features import featurize, action_key, INPUT_SIZE, ACTION_SPACE_SIZE

# 1. Initialize Firebase
# Ensure you have your serviceAccountKey.json
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

class GameLogDataset(Dataset):
    def __init__(self, games_collection="episodes"):
        self.samples = []
        self._load_from_firebase(games_collection)

    def _load_from_firebase(self, collection_name):
        print("Downloading logs from Firebase...")
        # Get all games
        games = db.collection(collection_name).stream()
        
        for game_doc in games:
            # For each game, get the steps
            steps = game_doc.reference.collection("steps").order_by("index").stream()
            
            # Simple RL logic: Winner gets +1, Loser gets -1
            # We need to look at the FINAL step to know who won
            steps_list = list(steps)
            if not steps_list: continue
            
            final_step = steps_list[-1].to_dict()
            # Assuming info['winner'] is stored in the last step
            winner = final_step.get("info", {}).get("winner") 
            
            for step in steps_list:
                data = step.to_dict()
                state = data['state']
                side = data['side']
                action = data['action']
                mcts_policy = data.get('policy')
                
                # --- Prepare Training Data ---
                # 1. Input: Vectorize the state
                # Note: We must know WHICH side's perspective to featurize
                input_vec = featurize(state, side)
                
                # 2. Policy Target: The MCTS probabilities
                policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                if mcts_policy:
                    for act_str, prob in mcts_policy.items():
                        try:
                            # We need the action ID. 
                            # WARNING: Ensure action_map is loaded/consistent!
                            act_dict = json.loads(act_str)
                            idx = action_key(act_dict)
                            policy_target[idx] = prob
                        except:
                            pass
                else:
                    # Fallback: One-hot encode the actual move taken
                    idx = action_key(action)
                    policy_target[idx] = 1.0

                # 3. Value Target: Did this side win?
                # If winner == side: +1, else -1 (simplified)
                if winner:
                    value_target = 1.0 if winner == side else -1.0
                else:
                    value_target = 0.0 # Draw or incomplete

                self.samples.append((input_vec, policy_target, value_target))
        print(f"Loaded {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 0.001

    # Load Data
    dataset = GameLogDataset()
    if len(dataset) == 0:
        print("No data found. Play some games first!")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Model (or create new)
    model = PVModel()
    if os.path.exists("israel_model.pth"):
        print("Loading existing brain to update...")
        model = PVModel.load("israel_model.pth")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, target_policies, target_values in loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_policies, pred_values = model(states)
            
            # Loss Calculation
            # 1. Value Loss (MSE): Predict who wins
            loss_v = ((pred_values.squeeze() - target_values.float()) ** 2).mean()
            
            # 2. Policy Loss (Cross Entropy): Predict MCTS move
            # Convert target probs to log-probs for KLDiv or similar
            # Simple approach: CrossEntropy expects class indices, but we have distributions.
            # We use manually calculated Cross Entropy for distributions:
            log_probs = torch.log_softmax(pred_policies, dim=1)
            loss_p = -(target_policies * log_probs).sum(dim=1).mean()
            
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # Save updated brain
    model.save("israel_model.pth")
    # Also save for Iran if you want a shared brain, or train separate model
    model.save("iran_model.pth") 
    print("✅ Brain updated! Restart the API to use new smarts.")

if __name__ == "__main__":
    from features import load_action_map
    # CRITICAL: Load the action map so IDs match what the API used
    load_action_map() 
    train()
