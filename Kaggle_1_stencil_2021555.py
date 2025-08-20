import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class CompetitionDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing competition data.
    Participants should modify this to load their specific dataset.
    """
    def __init__(self, data_path, dtype="train", *args, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the dataset
        """
        # TODO: Load your data here
        self.data = pd.read_csv(data_path)
        self.dtype = dtype
        if self.dtype == "test":
            self.ids = self.data['INDEX'].values
            self.features = self.data[['F1', 'F2', 'F3']].values  
        else:
            self.features = self.data[['F1', 'F2', 'F3']].values  
            self.labels = self.data['OUT'].values  
        
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        # Dummy data length
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Generate one sample of data.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (input data, label)
        """
        # TODO: Implement proper data and label extraction

        if self.dtype == "test":
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            return self.ids[idx], features
        else:
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features, label


class CompetitionModel(nn.Module):
    """
    Base neural network model for the competition.
    Participants should modify the architecture as needed.
    """
    def __init__(self, input_dim, hidden_dims=1, output_dim=1):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output
        """
        super().__init__()
        
        # Create layers dynamically based on hidden_dims
        self.wide = nn.Sequential(
            nn.Linear(input_dim, 1024),
            # nn.LayerNorm(128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(146, 292),
            # 
        )
        
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.SiLU()
            
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=16+1024, num_heads=80)
        
        self.final = nn.Sequential(
            nn.Linear(16+1024, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        wide = self.wide(x)
        deep = self.deep(x)
        combined = torch.cat([deep, wide], dim=1)
        
        attn_out, _ = self.attention(combined.unsqueeze(0), combined.unsqueeze(0), combined.unsqueeze(0))
        attn_out = attn_out.squeeze(0)
        
        return self.final(0.6 * combined + 0.4 * attn_out)

class CompetitionCriterion(nn.Module):
    """
    Custom loss function for the competition.
    Combines MSE with additional penalty or custom metrics.
    """
    def __init__(self, mse_weight=0.8, complexity_penalty=0.2):
        """
        Initialize the custom loss criterion.
        
        Args:
            mse_weight (float): Weight for Mean Squared Error
            complexity_penalty (float): Weight for model complexity penalty
        """
        super(CompetitionCriterion, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.complexity_penalty = complexity_penalty
    
    def forward(self, predictions, targets, model=None):
        """
        Compute the custom loss.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            model (nn.Module, optional): Neural network model for complexity calculation
        
        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate base Mean Squared Error
        mse = self.mse_loss(predictions, targets)
        
        # Optional model complexity penalty
        complexity_loss = 0
        if model is not None:
            # Example: L2 regularization (weight decay)
            for param in model.parameters():
                complexity_loss += torch.norm(param, p=2)
        
        # Combine losses
        total_loss = (self.mse_weight * mse) + (self.complexity_penalty * complexity_loss)
        
        return total_loss


def train_model(model, train_loader, criterion, optimizer, num_epochs=150):
    """
    Train the neural network model.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
    
    Returns:
        dict: Training history with train losses
    """
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
    best_loss = float('inf')
    best_model_path = 'competition_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    history = {
        'train_loss': []
    }
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * features.size(0)
            total_samples += features.size(0)

        epoch_loss = train_loss_total / total_samples
        scheduler.step(epoch_loss)
        
        history['train_loss'].append(epoch_loss)
        
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with Loss: {best_loss:.4f}')
    
    return history

def test_model(model_path, test_data_path, output_path='predictions.csv'):
    """
    Test a trained model and generate predictions in a specified CSV format.
    
    Args:
        model_path (str): Path to the saved model weights
        test_data_path (str): Path to the test dataset CSV
        output_path (str, optional): Path to save prediction results. Defaults to 'predictions.csv'
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model (replace MyModel with your actual model class)
    model = CompetitionModel(input_dim=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test data
    test_data = CompetitionDataset(test_data_path, dtype="test")
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    ids_all = []
    predictions = []
    
    with torch.no_grad():
        for ids, features in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            predictions.extend(outputs.cpu().numpy().tolist())
            if isinstance(ids, torch.Tensor):
                ids_all.extend(ids.cpu().numpy().tolist())
            else:
                ids_all.extend(ids)

    output_df = pd.DataFrame({
        "ID": ids_all,
        "OUT": predictions
    })

	# Write your code here
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(output_df)}")
    
    output_df.to_csv(output_path, index=False)
    
    return output_df


def main():
    """
    Main function to set up and run the training pipeline.
    Participants should customize this based on their specific requirements.
    """
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    # Data loading
    train_dataset = CompetitionDataset('kaggle_1_train.csv')  # Replace with actual path
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    input_dim = train_dataset.features.shape[1]
    hidden_dims = 1 # dummy variable not used
    output_dim = 1
    
    # Model initialization
    model = CompetitionModel(input_dim, hidden_dims, output_dim)
    
    # Loss and optimizer
    criterion = CompetitionCriterion()  # Competition Error
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Training
    history = train_model(model, train_loader, criterion, optimizer)
    
    test_model('competition_model.pth', 'kaggle_1_test.csv', )


if __name__ == '__main__':
    main()
