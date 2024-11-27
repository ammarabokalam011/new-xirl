# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import numpy as np
import random
import torch
from absl import app, flags
from torch_geometric.data import DataLoader
from xirl.models import GNNModel  # Assuming GNNModel is defined in xirl.models

FLAGS = flags.FLAGS

flags.DEFINE_string("graph_data_path", './data/graphs/combined_graph.pt', "Path to graph path.")
flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("restore_checkpoint", True, "Restore model checkpoint.")
flags.DEFINE_string("dataset_folder", './path_to_dataset', "Path to dataset folder containing video frames.")

def extract_features(img_file):
    """Extract features from an image file."""
    img = cv2.imread(img_file)
    img = cv2.resize(img, (128, 128))  # Resize to desired dimensions
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0  # Convert to tensor and normalize
    return img_tensor.view(-1).numpy()  # Flattening for simplicity

def get(image_folder):
    """Extract features from all images in the specified folder."""
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    features = []

    print(f"Extracting features from {len(image_files)} images in {image_folder}...")

    for img_file in image_files:
        feature_vector = extract_features(img_file)
        features.append(feature_vector)

    return np.array(features).squeeze()

def calculate_rewards(embeddings):
    """Calculate rewards based on embeddings."""
    rewards = [np.linalg.norm(emb) for emb in embeddings]  # Example reward calculation (can be modified)
    return np.array(rewards)

def increasing_reward_loss(rewards):
    """Calculate loss based on increasing reward condition."""
    differences = rewards[1:] - rewards[:-1]  # Differences between consecutive rewards
    loss = torch.sum(torch.relu(-differences))  # Penalize if any difference is negative (i.e., reward does not increase)
    return loss

def train(gnn_model, gnn_data_loader, optimizer, device):
    """Train the GNN model using graph data."""
    gnn_model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch in gnn_data_loader:
        optimizer.zero_grad()  # Clear gradients
        
        batch = batch.to(device)  # Move batch to device
        predictions = gnn_model(batch.x, batch.edge_index, batch.batch)  # Forward pass
        
        # Here you can define your ground truth labels if needed
        # loss = custom_loss_fn(predictions, batch.y) 
        loss = ...  # Define your loss calculation based on predictions and targets
        
        loss.backward()  # Backpropagation step
        optimizer.step()  # Update weights
        
        total_loss += loss.item()

    return total_loss / len(gnn_data_loader)

def setup(graph_data_path):
    """Load graph data and initialize GNN model."""
    graph_data = torch.load(graph_data_path)  # Load the graph data
    
    node_features = graph_data.x  # Node features tensor
    edge_index = graph_data.edge_index  # Edge index tensor
    
    gnn_model = GNNModel(input_dim=node_features.size(1), hidden_dim_1=2000, hidden_dim_2=265, output_dim=32) 
    gnn_data_loader = DataLoader([graph_data], batch_size=10)  # Create a DataLoader for the graph data

    return gnn_model, gnn_data_loader

def main(_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use CUDA if available
    
    gnn_model, gnn_data_loader = setup(FLAGS.graph_data_path)
    
    gnn_model.to(device)  # Move model to device
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)  # Initialize optimizer
    
    num_epochs = 10  # Set the number of epochs
    
    for epoch in range(num_epochs):
        avg_loss = train(gnn_model, gnn_data_loader, optimizer, device)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')  # Print average loss
        
        # After training with graph data, calculate loss based on video frames
        random_video_folder = random.choice(os.listdir(FLAGS.dataset_folder))  # Randomly select a video folder
        full_video_path = os.path.join(FLAGS.dataset_folder, random_video_folder)
        
        if os.path.isdir(full_video_path):
            features = get(full_video_path)  # Extract features from selected video folder
            
            embeddings = []
            for feature in features:
                feature_tensor = torch.tensor(feature).float().to(device)  # Convert to tensor
                
                with torch.no_grad():
                    emb = gnn_model(feature_tensor.unsqueeze(0)).cpu().numpy()  # Get embedding
                
                embeddings.append(emb)

            rewards = calculate_rewards(embeddings)  # Calculate rewards based on embeddings
            
            rewards_tensor = torch.tensor(rewards).to(device)  # Convert rewards to tensor for loss calculation
            
            loss_from_video_frames = increasing_reward_loss(rewards_tensor)  # Calculate loss based on increasing reward condition
            
            print(f'Loss from video frames after epoch {epoch + 1}: {loss_from_video_frames.item()}') 

if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    flags.mark_flag_as_required("dataset_folder")
    app.run(main)