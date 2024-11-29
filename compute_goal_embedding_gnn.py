import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from absl import app, flags
from torch_geometric.loader.dataloader import DataLoader
from xirl.models import GNNModel  # Assuming GNNModel is defined in xirl.models
from torch_geometric.data import Data
import logging
from utils import extract_features

FLAGS = flags.FLAGS

flags.DEFINE_string("graph_data_path", './data/graphs/combined_graph.pt', "Path to graph path.")
flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("restore_checkpoint", True, "Restore model checkpoint.")
flags.DEFINE_string("dataset_folder", None, "Path to dataset folder containing video frames.")

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        self.previous_reward = None  # To store the previous reward

    def forward(self, predictions, device):
        current_reward = self.calculate_reward(predictions, device)
        print('Current Reward:', current_reward.item())

        if self.previous_reward is None:
            self.previous_reward = torch.tensor(0.0, device=current_reward.device)  # Initialize as tensor

        # Calculate loss based on the difference in rewards
        loss = (current_reward - self.previous_reward).clamp(min=0)  # Ensure non-negative loss

        # Update previous reward
        self.previous_reward = current_reward.detach()  # Detach to avoid tracking gradients
        return loss  # Return the loss directly; it will automatically have requires_grad=True if computed from tensors

    def calculate_reward(self, predictions, device):
        """Calculate rewards based on distance between predictions and video frame features."""
        
        video_folder = random.choice(os.listdir(FLAGS.dataset_folder))
        video_features = self.get(os.path.join(FLAGS.dataset_folder, video_folder))

        predictions = predictions.to(device)

        if predictions.dim() == 2:  # If shape is [1, 32], expand it
            predictions = predictions.unsqueeze(1).expand(-1, len(video_features), -1)

        distances = []
        for i in range(len(video_features)):
            distance = np.linalg.norm(video_features[i] - predictions[0][i].detach().cpu().numpy())
            distances.append(max(0.0, distance))

        average_distance = np.mean(distances)
        
        return torch.tensor(max(0.0, average_distance), dtype=torch.float32, device=predictions.device)  # Ensure this is a tensor
    
    def get(self, image_folder):
        """Extract features from all images in the specified folder."""
        image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        features = []

        logging.info(f"Extracting features from {len(image_files)} images in {image_folder}...")

        for img_file in image_files:
            feature_vector = extract_features(img_file)
            features.append(feature_vector)

        return np.array(features).squeeze()
    
def train(gnn_model, gnn_data_loader, optimizer, device):
    """Train the GNN model using graph data."""
    gnn_model.train()  # Set the model to training mode
    total_loss = 0
    
    reward_loss_fn = RewardLoss()  # Instantiate RewardLoss
    
    for batch in gnn_data_loader:
        # optimizer.zero_grad()  # Clear gradients
        
        batch = batch.to(device)  # Move batch to device
        predictions = gnn_model(batch.x, batch.edge_index, batch.batch)  # Forward pass
        logging.info(f'predictions: {predictions}')
        
        loss = reward_loss_fn(predictions, device)  # Calculate custom loss
        
        print('Loss:', loss.item())  # Print loss for debugging
        
        # if loss.requires_grad:  # Ensure that loss requires gradient tracking
        loss.requires_grad = True

        loss.backward()  # Backpropagation step
        
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
        
    return total_loss / len(gnn_data_loader)

def setup(graph_data_path):
    """Load graph data and initialize GNN model."""
    graph_data = torch.load(graph_data_path)  # Load the graph data
    
    node_features = graph_data.x  # Node features tensor
    edge_index = graph_data.edge_index  # Edge index tensor
    
    gnn_model = GNNModel(input_dim=node_features.size(1), hidden_dim=1000, output_dim=32) 
    gnn_data_loader = DataLoader([graph_data], batch_size=1)  # Create a DataLoader for the graph data

    return gnn_model, gnn_data_loader

def main(_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use CUDA if available

    logging.basicConfig(level=logging.INFO)
    gnn_model, gnn_data_loader = setup(FLAGS.graph_data_path)
    
    gnn_model.to(device)  # Move model to device
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.1)  # Initialize optimizer
    
    num_epochs = 3000  # Set the number of epochs
    
    for epoch in range(num_epochs):
        avg_loss = train(gnn_model, gnn_data_loader, optimizer, device)
        
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')  # Print average loss
        
if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    flags.mark_flag_as_required("dataset_folder")
    app.run(main)