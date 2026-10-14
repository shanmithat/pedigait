"""
Temporal CNN for Gait Classification Module

Implements temporal convolutional neural network for classifying gait patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. CNN classification disabled.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ClassificationResult:
    """Container for classification results."""
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    accuracy: float = 0.0
    f1_score: float = 0.0
    class_distribution: Optional[Dict] = None


class GaitDataset(Dataset):
    """PyTorch Dataset for gait data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize gait dataset.
        
        Args:
            features: Array of shape (n_samples, n_timesteps, n_features)
            labels: Array of shape (n_samples,) with class labels
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.long)


class TemporalCNN(nn.Module):
    """Temporal CNN for gait classification."""
    
    def __init__(self, n_classes: int = 5, n_features: int = 18, 
                 n_timesteps: int = 100, hidden_dim: int = 128):
        """
        Initialize temporal CNN.
        
        Args:
            n_classes: Number of output classes
            n_features: Number of input features
            n_timesteps: Number of timesteps
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Temporal convolution layers
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc_out = nn.Linear(64, n_classes)
    
    def forward(self, x: torch.Tensor, anthropometry: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, timesteps, features)
            anthropometry: Optional anthropometric features
            
        Returns:
            Output tensor of shape (batch, n_classes)
        """
        # Temporal convolutions
        h = self.conv1(x.transpose(1, 2))
        h = self.drop1(self.act1(self.bn1(h))).transpose(1, 2)
        
        h = self.conv2(h.transpose(1, 2))
        h = self.drop2(self.act2(self.bn2(h))).transpose(1, 2)
        
        # Bidirectional LSTM
        h, _ = self.bilstm(h)
        
        # Take the last timestep output
        h = h[:, -1, :]
        
        # Fully connected
        h = self.drop3(self.act3(self.fc1(h)))
        
        # Output
        return self.fc_out(h)


class GaitClassifier:
    """Gait classification using temporal CNN."""
    
    # Class names for KSwS classification
    CLASS_NAMES = [
        "Normal",
        "Mild",
        "Moderate",
        "Severe",
        "Highly Pathological"
    ]
    
    def __init__(self, n_classes: int = 5, device: str = None):
        """
        Initialize gait classifier.
        
        Args:
            n_classes: Number of output classes
            device: Device to use (cpu or cuda)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this classifier")
        
        self.n_classes = n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = self.CLASS_NAMES[:n_classes]
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001) -> Dict:
        """
        Train the classifier.
        
        Args:
            X_train: Training features (n_samples, n_timesteps, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        # Initialize model
        self.model = TemporalCNN(
            n_classes=self.n_classes,
            n_features=X_train.shape[2],
            n_timesteps=X_train.shape[1]
        ).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = GaitDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = GaitDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Train: loss={train_loss:.4f}, acc={train_acc:.4f}"
                if val_loader is not None:
                    msg += f" - Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
                print(msg)
        
        return history
    
    def predict(self, X: np.ndarray) -> ClassificationResult:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, n_timesteps, n_features)
            
        Returns:
            ClassificationResult object
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, predicted) if hasattr(self, 'y_true') else 0.0
        f1 = f1_score(self.y_true, predicted, average='weighted') if hasattr(self, 'y_true') else 0.0
        
        return ClassificationResult(
            predictions=predicted,
            true_labels=getattr(self, 'y_true', None),
            probabilities=probabilities,
            accuracy=accuracy,
            f1_score=f1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier (train).
        
        Args:
            X: Training features
            y: Training labels
        """
        # Store labels for evaluation
        self.y_true = y
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        history = self.train(X_train, y_train, X_val, y_val, epochs=100)
        
        return history
    
    def save(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        self.model = TemporalCNN(
            n_classes=self.n_classes,
            n_features=64,  # Will be updated based on input
            n_timesteps=100
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def classify_ksws(self, kinematic_features: np.ndarray) -> List[str]:
        """
        Classify KSwS index from kinematic features.
        
        Args:
            kinematic_features: Array of kinematic features (n_samples, n_timesteps, n_features)
            
        Returns:
            List of KSwS classifications
        """
        result = self.predict(kinematic_features)
        
        # Convert numeric predictions to class names
        classifications = [self.class_names[p] for p in result.predictions]
        
        return classifications


class TabularClassifier:
    """Tabular baseline classifier for comparison."""
    
    def __init__(self):
        """Initialize tabular classifier."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> ClassificationResult:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            ClassificationResult object
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return ClassificationResult(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the classifier.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        result = self.predict(X)
        
        return {
            "accuracy": accuracy_score(y, result.predictions),
            "f1_score": f1_score(y, result.predictions, average='weighted')
        }


class FeatureExtractor:
    """Extract features from kinematic data for classification."""
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(self, kinematic_data) -> np.ndarray:
        """
        Extract classification features from kinematic data.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Feature array
        """
        features = []
        
        # Extract features from each joint
        for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                          "knee_right", "ankle_left", "ankle_right"]:
            joint = getattr(kinematic_data, joint_name)
            
            # Calculate basic statistics for each angle component
            for i in range(3):  # Sagittal, frontal, transverse
                component = joint.angles[:, i]
                
                features.extend([
                    np.min(component),
                    np.max(component),
                    np.mean(component),
                    np.std(component),
                    np.median(component),
                ])
        
        # Add symmetry features
        symmetry_analyzer = SymmetryAnalyzer()
        symmetry_results = symmetry_analyzer.analyze_all_joints_symmetry(kinematic_data)
        
        for sym in symmetry_results.values():
            features.extend([
                sym.symmetry_index,
                sym.asymmetry_score,
                sym.correlation,
                sym.mean_difference,
                sym.std_difference
            ])
        
        return np.array(features)
    
    def extract_temporal_features(self, kinematic_data, 
                                  n_timesteps: int = 100) -> np.ndarray:
        """
        Extract temporal features for CNN classification.
        
        Args:
            kinematic_data: Processed kinematic data
            n_timesteps: Number of timesteps to use
            
        Returns:
            Temporal feature array (n_timesteps, n_features)
        """
        # Resample to consistent timesteps
        n_frames = kinematic_data.knee_left.angles.shape[0]
        
        if n_frames < n_timesteps:
            # Pad with zeros
            pad_length = n_timesteps - n_frames
            features = np.zeros((n_timesteps, 18))
            
            # Fill with available data
            for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                              "knee_right", "ankle_left"]:
                joint = getattr(kinematic_data, joint_name)
                for i in range(3):
                    col_idx = ["pelvis", "hip_left", "hip_right", "knee_left", 
                              "knee_right", "ankle_left"].index(joint_name) * 3 + i
                    features[:n_frames, col_idx] = joint.angles[:, i]
        else:
            # Downsample
            indices = np.linspace(0, n_frames - 1, n_timesteps).astype(int)
            features = np.zeros((n_timesteps, 18))
            
            for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                              "knee_right", "ankle_left"]:
                joint = getattr(kinematic_data, joint_name)
                for i in range(3):
                    col_idx = ["pelvis", "hip_left", "hip_right", "knee_left", 
                              "knee_right", "ankle_left"].index(joint_name) * 3 + i
                    features[:, col_idx] = joint.angles[indices, i]
        
        return features


# Import SymmetryAnalyzer for feature extraction
try:
    from ..analysis.symmetry_analysis import SymmetryAnalyzer
except ImportError:
    # Fallback for testing
    class SymmetryAnalyzer:
        def analyze_all_joints_symmetry(self, kinematic_data):
            return {}
