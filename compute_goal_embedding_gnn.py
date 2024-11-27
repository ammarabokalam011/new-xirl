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
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

FLAGS = flags.FLAGS

flags.DEFINE_string("graph_data_path", './data/graphs/combined_graph.pt', "Path to graph path.")
flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("restore_checkpoint", True, "Restore model checkpoint.")
flags.DEFINE_string("dataset_folder", './path_to_dataset', "Path to dataset folder containing video frames.")

def extract_features(image_path):
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        features = resnet(img_tensor)

    return features.view(features.size(0), -1).numpy()

def get(image_folder):
    """Extract features from all images in the specified folder."""
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    features = []

    print(f"Extracting features from {len(image_files)} images in {image_folder}...")

    for img_file in image_files:
        feature_vector = extract_features(img_file)
        features.append(feature_vector)

    return np.array(features).squeeze()

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

        
if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    flags.mark_flag_as_required("dataset_folder")
    app.run(main)