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

"""Compute and store the mean goal embedding using a trained model."""

import os
import typing

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import utils
from xirl import common
from xirl.models import SelfSupervisedModel
from xirl.models import GNNModel
from torch_geometric.data import DataLoader
import random

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("graph_data_path", './data/graphs/combined_graph.pt', "Path to graph path.")
flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")



def emb(model, data_loader, device):
    model.eval()
    gnn_embeddings = []
    init_embs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing embeddings", unit="batch"):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_indices = batch.batch.to(device)

            out = model(x, edge_index)  # Use forward pass of LargeScaleGNN
            gnn_embeddings.append(out.cpu().numpy())
            init_embs.append(out[0].cpu().numpy())

    gnn_embeddings = np.concatenate(gnn_embeddings)
    mean_embedding = np.mean(gnn_embeddings, axis=0, keepdims=True)
    dist_to_mean = np.linalg.norm(init_embs - mean_embedding, axis=-1).mean()
    distance_scale = 1.0 / dist_to_mean if dist_to_mean != 0 else float('inf')
    
    return mean_embedding, distance_scale

def setup(graph_data_path):
    # Load the graph data
    graph_data = torch.load(graph_data_path)

    # Access node features and edge indices
    node_features = graph_data.x  # Node features tensor
    edge_index = graph_data.edge_index  # Edge index tensor

    # Initialize GNN model
    gnn_model = GNNModel(input_dim=node_features.size(1), hidden_dim_1=3000, hidden_dim_2=265,output_dim=32)

    # Create a DataLoader for the graph data (assuming it's a single graph)
    gnn_data_loader = DataLoader([graph_data], batch_size=1)  # Adjust as needed

    return gnn_model, gnn_data_loader

def main(_):
    device = "cuda:0"  # Use CUDA if available
    gnn_model, gnn_data_loader = setup(FLAGS.graph_data_path)
    gnn_model.to(device).eval()

    # Clear cache before embedding
    torch.cuda.empty_cache()
    gnn_embeddings, distance_scale_gnn = emb(gnn_model, gnn_data_loader, device)
    gnn_embeddings = gnn_embeddings * 10
    # Save individual embeddings and distance scale to separate files
    utils.save_pickle(FLAGS.experiment_path, gnn_embeddings, "gnn_emb.pkl")
    utils.save_pickle(FLAGS.experiment_path, distance_scale_gnn, "gnn_distance_scale.pkl")

    # Clear cache before embedding
    torch.cuda.empty_cache()

if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    # flags.mark_flag_as_required("graph_data_path")
    app.run(main)