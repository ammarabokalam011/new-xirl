import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from absl import app, flags
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
import logging
from xirl.models import GNNModel  # Assuming GNNModel is defined in xirl.models
from utils import extract_features
import utils
FLAGS = flags.FLAGS

flags.DEFINE_string("graph_data_path", './data/graphs/combined_graph.pt', "Path to graph path.")
flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("restore_checkpoint", True, "Restore model checkpoint.")
flags.DEFINE_string("dataset_folder", None, "Path to dataset folder containing video frames.")

def get( image_folder):
    """Extract features from all images in the specified folder."""
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    features = []

    logging.info(f"Extracting features from {len(image_files)} images in {image_folder}...")

    for img_file in image_files:
        feature_vector = extract_features(img_file)
        features.append(feature_vector)

    return np.array(features).squeeze()

def get_distances(predictions,video_features): 
    distances = []
    for i in range(len(video_features)):
        distance = np.linalg.norm(video_features[i] - predictions[0])
        distances.append(max(0.0, distance))
    return distances

def train(gnn_model, gnn_data_loader, optimizer, device):
    """Train the GNN model using graph data."""
    gnn_model.train()  # Set the model to training mode
    total_loss = 0
    
    # reward_loss_fn = RewardLoss()  # Instantiate RewardLoss
    gnn_embeddings = []
    init_embs = []

    for batch in gnn_data_loader:
        optimizer.zero_grad()  # Clear gradients
        torch.no_grad()
        batch = batch.to(device)  # Move batch to device
        predictions = gnn_model(batch.x, batch.edge_index, batch.batch)  # Forward pass
        logging.info(f'Predictions: {predictions}')  # Debugging output
        # loss = reward_loss_fn(predictions, device)  # Calculate custom loss using RewardLoss
        # loss.requires_grad = True
        # logging.info(f'Loss: {loss}')  # Print loss for debugging
        video_folder = random.choice(os.listdir(FLAGS.dataset_folder))
        video_features = get(os.path.join(FLAGS.dataset_folder, video_folder))
        # logging.info(f'video_features: {video_features}')  # Debugging output
        logging.info(f'video_features shape: {video_features.shape}, Predictions: {predictions.shape}')  # Debugging output
        video_features = torch.tensor(video_features, device=device)
        loss = nn.MSELoss()
        # predictions_rep = predictions.repeat(video_features.shape[0], 1)
        
        logging.info(f'predictions size: {predictions.size}, video_features size: {video_features[-1].size}')  # Debugging output
        logging.info(f'type(size) : {type(video_features[-1].size)}, type(size) :{type(predictions.size)}')
        goal = torch.tensor(video_features[-1], requires_grad=True, device=device)
        # predictions = predictions.unsqueeze(1)
        
        output = loss(predictions,goal)
        output.backward()
        
        optimizer.step()  # Update weights
            
        gnn_embeddings.append(predictions.detach().cpu().numpy())  
            # Capture the first node's embedding for distance calculation
        init_embs.append(predictions.detach().cpu().numpy()[0])    
        total_loss += output
    # Concatenate all embeddings
    
    gnn_embeddings = np.concatenate(gnn_embeddings)

    # Calculate mean embedding across all nodes
    mean_embedding = np.mean(gnn_embeddings, axis=0, keepdims=True)

    # Calculate distance to mean embedding
    dist_to_mean = np.linalg.norm(init_embs - mean_embedding, axis=-1).mean()

    if dist_to_mean < 1e-6:
        distance_scale = 1.0  # or some predefined maximum scale value
    else:
        distance_scale = 1.0 / dist_to_mean
    return gnn_embeddings, distance_scale, total_loss / len(gnn_data_loader)


def setup(graph_data_path):
    """Load graph data and initialize GNN model."""
    graph_data = torch.load(graph_data_path)  # Load the graph data
    
    node_features = graph_data.x  # Node features tensor
    edge_index = graph_data.edge_index  # Edge index tensor
    
    logging.info(f'Node Features Shape: {node_features.shape}, Edge Index Shape: {edge_index.shape}')
    
    gnn_model = GNNModel(input_dim=node_features.size(1), hidden_dim1=1000, hidden_dim2=250,output_dim=32) 
    gnn_data_loader = DataLoader([graph_data], batch_size=1)  # Create a DataLoader for the graph data

    return gnn_model, gnn_data_loader

def main(_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use CUDA if available

    logging.basicConfig(level=logging.INFO)
    gnn_model, gnn_data_loader = setup(FLAGS.graph_data_path)
    
    gnn_model.to(device)  # Move model to device
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)  # Initialize optimizer
    
    num_epochs = 30  # Set the number of epochs
    
    for epoch in range(num_epochs):
        gnn_embeddings, distance_scale,avg_loss = train(gnn_model, gnn_data_loader, optimizer, device)
        
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')  # Print average loss
        logging.info(f'Epoch {epoch + 1} - Embeddings shape: {gnn_embeddings.shape}, Distance Scale: {distance_scale:.4f}')

    # Clear cache before embedding
    torch.cuda.empty_cache()
    
	# Save individual embeddings and distance scale to separate files
    utils.save_pickle(FLAGS.experiment_path, gnn_embeddings, "gnn_emb.pkl")
    utils.save_pickle(FLAGS.experiment_path, distance_scale, "gnn_distance_scale.pkl")
    
	# Clear cache before embedding
    torch.cuda.empty_cache()

if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    flags.mark_flag_as_required("dataset_folder")
    app.run(main)