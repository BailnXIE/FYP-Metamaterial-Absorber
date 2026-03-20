"""
==========================================
Guide to Improving Model Prediction Accuracy
==========================================

This model provides several configurable accuracy enhancement options:

1. **Model Architecture Improvements:**
- **USE_DEEPER_NETWORK:** Use a deeper network (Increase the number of layers and residual blocks)
- **USE_WIDER_NETWORK:** Use a wider network (increase the width of each layer)
→ Improves model expressiveness, but requires more computational resources.

2. **Training Strategy Improvements:**
- **INCREASED_EPOCHS:** Increase the number of training epochs (3000 vs 2000)
- **INCREASED_PATIENCE:** Increase the early stopping patience value (150 vs 100)
- **BETTER_LOSS_WEIGHTS:** Optimize the loss function weights
→ Give the model more time to converge, using a better loss function.

3. **Learning Rate Strategy:**
- **LR_SCHEDULER_TYPE:** Choose 'cosine', 'plateau', 'step', 'onecycle'
→ Different learning rate strategies may be more effective for different datasets.

4. **Data Augmentation:**
- USE_DATA_AUGMENTATION: Add slight noise (use with caution)
→ May improve generalization ability, but use with caution for regression problems

5. [Parameter Precision]
- DECIMAL_PLACES: Adjust parameter precision (3, 4, 5 decimal places)
→ Higher precision may improve model performance, but may also increase the risk of overfitting

6. [Other Suggestions]
- Increase the amount of training data (most effective method)
- Try ensemble learning (train multiple models and average the predictions)
- Adjust regularization parameters (weight_decay, dropout)
- Use cross-validation to select the optimal hyperparameters

==========================================
"""

import os
# repairing the OpenMP conflix on macOS 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR

# ==========================================
# Device detection (Mac GPU acceleration support)
# ==========================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅  Mac GPU (MPS) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU only（if there is no GPU detected）")
    return device

device = get_device()

# ==========================================
# 1. Define the dataset (processing CSV)
# ==========================================
class MetaSurfaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Improved neural network model (with residual connections and feature interactions)
# ==========================================
class ResidualBlock(nn.Module):
    """Residual blocks help gradient flow"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # Using LayerNorm instead of BatchNorm, it is more stable for small datasets.
            nn.SiLU(),  # The SiLU activation function is smoother than ReLU.
            nn.Dropout(0.15),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(self.block(x) + x)  # residual connection

class AbsorberNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 512, 512, 256, 128]):
        super(AbsorberNet, self).__init__()
        
        # input layer
        layers = []
        prev_dim = input_dim
        
        # Build the main network
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if i < len(hidden_dims) - 1:  # The last layer does not have Dropout.
                layers.append(nn.Dropout(0.15))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Residual blocks (enhanced feature representation)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dims[-1]),
            ResidualBlock(hidden_dims[-1]),
        )
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()  # Absorption rate between 0 and 1
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.residual_blocks(features)
        return self.output_layer(features)

# ==========================================
# 3. Data preprocessing and loading (key: precision handling)
# ==========================================
print("--- Reading data (Rings structure) ---")
df = pd.read_csv('./data/ring_data.csv')

# Define input parameters and output target 
# (Rings: excluding t_g, t_g is the underlying layer and mainly affects transmission with little help to absorption).
feature_cols = ['resolution', 'period', 'n_rings', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8', 'r_9', 'r_10', 't_r', 't_s']
target_cols = [col for col in df.columns if col.startswith('Abs')]
FIXED_T_G = float(df['t_g'].mean())  # t_g is fixed as the dataset mean and is not used during optimization.

# Handle null values ​​from r_1 to r_10 (fill with 0).
for col in ['r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8', 'r_9', 'r_10']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Check the actual precision of the data (to help determine how many decimal places to use).
def check_data_precision(data, cols, name="data"):
    """Check the actual effective accuracy of the data"""
    print(f"\n--- checking the actual precision of {name} ---")
    for i, col in enumerate(cols):
        values = data[:, i] if len(data.shape) > 1 else data
        non_zero = values[values != 0]
        if len(non_zero) > 0:
            # Calculate the number of decimal places for non-zero values
            decimals = []
            for v in non_zero[:100]:  # Only the first 100 samples are checked to speed up the process.
                s = f"{v:.10f}".rstrip('0')
                if '.' in s:
                    decimals.append(len(s.split('.')[1]))
            if decimals:
                avg_decimals = np.mean(decimals)
                max_decimals = np.max(decimals)
                print(f"  {col:>10}: Average {avg_decimals:.1f} decimal places, Maximum {max_decimals:.0f} decimal places")
    print()

# Check the accuracy of the raw data
X_raw = df[feature_cols].values
check_data_precision(X_raw, feature_cols, "raw data")

# Configurable precision settings (can be changed to 3 or 5)
DECIMAL_PLACES = 3 
MIN_UNIT = 10 ** (-DECIMAL_PLACES)  # Minimum precision unit

# Calculate EPSILON dynamically based on precision (Recommended: EPSILON = MIN_UNIT * 20 - 50)
# For five decimal places: MIN_UNIT = 0.00001, EPSILON should be between 0.0002 and 0.0005
# For three decimal places: MIN_UNIT = 0.001, EPSILON should be between 0.002 and 0.005
EPSILON_MULTIPLIER = 50  # This multiplier can be adjusted (between 20 and 50).
EPSILON = MIN_UNIT * EPSILON_MULTIPLIER

print(f"--- Use {DECIMAL_PLACES} decimal places (minimum unit: {MIN_UNIT}, EPSILON: {EPSILON:.6f})---")
print(f"💡 Tip: If the effect is not good, you can try：")
print(f"   - Reduce precision to 3 decimal places（DECIMAL_PLACES = 3）")
print(f"   - Increase EPSILON ratio（EPSILON_MULTIPLIER = 50）")

# Round the parameter to the specified precision.
X_rounded = np.round(X_raw, DECIMAL_PLACES)
y = df[target_cols].values

# Check if the constraints are satisfied.（Rings: radii r_i > 0, r_1 >= r_2 >= ... >= r_n）
print("--- Check the constraints of the training data (Rings) ---")
constraint_violations = 0
idx_n_rings = feature_cols.index('n_rings')
idx_r1 = feature_cols.index('r_1')
for i in range(len(X_rounded)):
    n_rings = int(round(X_rounded[i, idx_n_rings]))
    radii = [X_rounded[i, idx_r1 + j] for j in range(min(n_rings, 10))]
    if n_rings > 0 and any(r <= 0 for r in radii):
        constraint_violations += 1
    if len(radii) > 1 and any(radii[j] < radii[j+1] for j in range(len(radii)-1)):
        constraint_violations += 1  # should satisfy r_1 >= r_2 >= ...
print(f"Number of data points that violated constraints: {constraint_violations} / {len(X_rounded)}")

# Standardization (after rounding)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rounded)

# Split the training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_dataset = MetaSurfaceDataset(X_train, y_train)
val_dataset = MetaSurfaceDataset(X_val, y_val)

# Use a smaller batch size because the amount of data is not large.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ==========================================
# 4. Improved training strategies (including multiple learning rate strategies)
# ==========================================
input_dim = len(feature_cols)
output_dim = len(target_cols)

# ==========================================
# Precision Enhancement Configuration Options
# ==========================================
# 1. Model architecture options (for 500 data points, a smaller network is recommended to reduce overfitting)
USE_DEEPER_NETWORK = False
USE_WIDER_NETWORK = False

if USE_DEEPER_NETWORK:
    hidden_dims = [256, 512, 512, 512, 256, 256, 128]
    num_residual_blocks = 3
elif USE_WIDER_NETWORK:
    hidden_dims = [512, 1024, 1024, 512, 256]
    num_residual_blocks = 2
else:
    # Simplified network: 500 data entries with a smaller capacity
    hidden_dims = [128, 256, 256, 128]  # Reduce layer width
    num_residual_blocks = 1              # Reduce residual blocks

# 2. Training strategy options (val): 
# If plateaued, early stopping will automatically terminate the training; the upper limit can be appropriately increased.
EPOCHS = 1500   # Maximum number of Epochs
INCREASED_EPOCHS = True

INCREASED_PATIENCE = True
BETTER_LOSS_WEIGHTS = True  # Use better loss function weights

# 3. Data augmentation options (slight noise helps mitigate overfitting)
USE_DATA_AUGMENTATION = True   # Enable slight noise to improve generalization

print(f"📊 Model configuration:")
print(f"   - Network depth: {len(hidden_dims)} ")
print(f"   - Number of residual blocks: {num_residual_blocks}")
print(f"   - data augmentation: {'enable' if USE_DATA_AUGMENTATION else 'Disable'}")

# Modify the model class to support a configurable number of residual blocks.
class AbsorberNetEnhanced(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 512, 512, 256, 128], num_residual_blocks=2):
        super(AbsorberNetEnhanced, self).__init__()
        
        # input layer
        layers = []
        prev_dim = input_dim
        
        # Build the main network
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if i < len(hidden_dims) - 1:  # No dropout on the last layer
                layers.append(nn.Dropout(0.2))  # Increase dropout to reduce overfitting
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Residual blocks (enhanced feature representation) - configurable count
        residual_blocks_list = []
        for _ in range(num_residual_blocks):
            residual_blocks_list.append(ResidualBlock(hidden_dims[-1]))
        self.residual_blocks = nn.Sequential(*residual_blocks_list)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()  # Absorption rate between 0 and 1
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.residual_blocks(features)
        return self.output_layer(features)

model = AbsorberNetEnhanced(input_dim, output_dim, hidden_dims=hidden_dims, num_residual_blocks=num_residual_blocks)
model = model.to(device)  # Move the model to the device

# Combined loss function: MSE + MAE + Huber Loss (configurable weights)
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=0.5, huber_weight=0.3, huber_delta=0.1):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        # Weighted combination
        return self.mse_weight * mse_loss + self.mae_weight * mae_loss + self.huber_weight * huber_loss

# Select loss function weights based on configuration
if BETTER_LOSS_WEIGHTS:
    # Emphasize MSE more (more sensitive to large errors) to improve accuracy
    criterion = CombinedLoss(mse_weight=1.0, mae_weight=0.3, huber_weight=0.2, huber_delta=0.05)
    print(f"📊 Using optimized loss function weights")
else:
    criterion = CombinedLoss()
    
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.9, 0.999))

# ==========================================
# Learning rate scheduler configuration
# ==========================================
# Available strategies: 'plateau', 'cosine', 'step', 'onecycle'
LR_SCHEDULER_TYPE = 'cosine'  # Change this to switch strategies: 'plateau', 'cosine', 'step', 'onecycle'

# Adjust training parameters based on configuration
# Use EPOCHS if set; otherwise use the INCREASED_EPOCHS logic
if EPOCHS is not None:
    epochs = EPOCHS
elif INCREASED_EPOCHS:
    epochs = 5000  # Increase training epochs
else:
    epochs = 2000

print(f"📊 Training configuration: Epochs = {epochs}")

if INCREASED_PATIENCE:
    patience = 50   # Stop if validation loss doesn't improve for 50 epochs to avoid later overfitting spikes
else:
    patience = 30

best_loss = float('inf')
patience_counter = 0

# Create the corresponding learning rate scheduler based on the selected strategy
scheduler_needs_metric = False  # By default, no validation loss is needed
scheduler_step_per_batch = False  # By default, step after each epoch

if LR_SCHEDULER_TYPE == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    scheduler_needs_metric = True  # Requires passing validation loss
    print(f"📊 Using LR scheduler: ReduceLROnPlateau (halve when validation loss stops improving)")
elif LR_SCHEDULER_TYPE == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler_needs_metric = False  # Does not require validation loss
    print(f"📊 Using LR scheduler: CosineAnnealingLR (cosine annealing, smooth decay)")
elif LR_SCHEDULER_TYPE == 'step':
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler_needs_metric = False
    print(f"📊 Using LR scheduler: StepLR (halve every 100 epochs)")
elif LR_SCHEDULER_TYPE == 'onecycle':
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, 
                          steps_per_epoch=len(train_loader))
    scheduler_needs_metric = False
    scheduler_step_per_batch = True  # OneCycleLR needs to be stepped after every batch
    print(f"📊 Using LR scheduler: OneCycleLR (single cycle, increase then decrease)")
else:
    raise ValueError(f"Unknown learning rate strategy: {LR_SCHEDULER_TYPE}")

# Record learning rate history (for visualization)
lr_history = []

print(f"--- Start training (Input: {input_dim} -> Output: {output_dim}) ---")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

loss_history = []
val_loss_history = []
# Track additional evaluation metrics
train_mae_history = []
val_mae_history = []
train_rmse_history = []
val_rmse_history = []

for epoch in range(epochs):
    # Training stage
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move to device
        
        # Data augmentation: add slight Gaussian noise (optional, only during training)
        if USE_DATA_AUGMENTATION and model.training:
            # Add slight noise (0.3% of the standard deviation) to improve generalization and reduce overfitting
            noise_scale = 0.003
            noise = torch.randn_like(inputs) * noise_scale
            inputs = inputs + noise
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        
        # OneCycleLR needs to update the learning rate after every batch
        if LR_SCHEDULER_TYPE == 'onecycle':
            scheduler.step()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation stage
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_mse = 0.0
    val_count = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Compute additional evaluation metrics
            mae = torch.mean(torch.abs(outputs - targets)).item()
            mse = torch.mean((outputs - targets) ** 2).item()
            val_mae += mae * len(inputs)
            val_mse += mse * len(inputs)
            val_count += len(inputs)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / val_count
    avg_val_rmse = np.sqrt(val_mse / val_count)
    
    # Compute training-set metrics (for comparison)
    model.eval()
    train_mae = 0.0
    train_mse = 0.0
    train_count = 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mae = torch.mean(torch.abs(outputs - targets)).item()
            mse = torch.mean((outputs - targets) ** 2).item()
            train_mae += mae * len(inputs)
            train_mse += mse * len(inputs)
            train_count += len(inputs)
    
    avg_train_mae = train_mae / train_count
    avg_train_rmse = np.sqrt(train_mse / train_count)
    
    loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    train_mae_history.append(avg_train_mae)
    val_mae_history.append(avg_val_mae)
    train_rmse_history.append(avg_train_rmse)
    val_rmse_history.append(avg_val_rmse)
    
    # Learning rate scheduling (based on different strategies)
    old_lr = optimizer.param_groups[0]['lr']
    
    if LR_SCHEDULER_TYPE == 'onecycle':
        # OneCycleLR is already updated after each batch; no need to update again here
        pass
    elif scheduler_needs_metric:
        # ReduceLROnPlateau requires passing in the validation loss
        scheduler.step(avg_val_loss)
    else:
        # CosineAnnealingLR and StepLR only need to be stepped after each epoch
        scheduler.step()
    
    new_lr = optimizer.param_groups[0]['lr']
    lr_history.append(new_lr)
    
    if old_lr != new_lr:
        print(f"  ⚠️  Learning rate changed: {old_lr:.2e} -> {new_lr:.2e}")
    
    # Early stopping mechanism
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        model_filename = f'/Users/xiebailin/meep_projects/pytorch/FYP/best_ring_{LR_SCHEDULER_TYPE}.pth'
        torch.save(model.state_dict(), model_filename)
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch+1) % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"Train MAE: {avg_train_mae:.6f}, Val MAE: {avg_val_mae:.6f}, "
              f"Train RMSE: {avg_train_rmse:.6f}, Val RMSE: {avg_val_rmse:.6f}, LR: {current_lr:.2e}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("✅ Model training complete!")

# Plot training curves (including learning rate and metric curves)
plt.figure(figsize=(20, 5))

# Subplot 1: Loss curve
plt.subplot(1, 4, 1)
plt.plot(loss_history, label='Train Loss', linewidth=2)
plt.plot(val_loss_history, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training History\n(LR Strategy: {LR_SCHEDULER_TYPE})')
plt.grid(True, alpha=0.3)

# Subplot 2: MAE curve (mean absolute error; lower is better)
plt.subplot(1, 4, 2)
plt.plot(train_mae_history, label='Train MAE', linewidth=2, color='blue', alpha=0.7)
plt.plot(val_mae_history, label='Val MAE', linewidth=2, color='orange', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error')
plt.grid(True, alpha=0.3)

# Subplot 3: RMSE curve (root mean squared error; lower is better)
plt.subplot(1, 4, 3)
plt.plot(train_rmse_history, label='Train RMSE', linewidth=2, color='blue', alpha=0.7)
plt.plot(val_rmse_history, label='Val RMSE', linewidth=2, color='orange', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.title('Root Mean Square Error')
plt.grid(True, alpha=0.3)

# Subplot 4: Learning rate curve
plt.subplot(1, 4, 4)
plt.plot(lr_history, label='Learning Rate', color='green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedule')
plt.yscale('log')  # Use a logarithmic scale to better observe changes
plt.grid(True, alpha=0.3)

# ==========================================
# 5. [Core Goal] Inverse optimization (with strict physical constraints)
# ==========================================
print("\n--- Running inverse optimization (with strict physical constraints) ---")

# Load the best model
model_filename = f'/Users/xiebailin/meep_projects/pytorch/FYP/best_ring_{LR_SCHEDULER_TYPE}.pth'
# if not os.path.exists(model_filename):
#     raise FileNotFoundError(
#         f"Model file not found: {model_filename}\n"
#         "Please run this script fully to complete training first; during training the best model is saved automatically."
#     )
model.load_state_dict(torch.load(model_filename))
model.eval()

# Define parameter indices (Rings: resolution, period, n_rings, r_1~r_10, t_r, t_s; t_g is fixed)
idx_resolution = 0
idx_period = 1
idx_n_rings = 2
idx_r1 = 3
idx_r2 = 4
idx_r3 = 5
idx_r4 = 6
idx_r5 = 7
idx_r6 = 8
idx_r7 = 9
idx_r8 = 10
idx_r9 = 11
idx_t_r = 13
idx_t_s = 14

# Prepare scaler parameters (move to device)
scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)

# Lock resolution to the highest precision
LOCKED_RESOLUTION_REAL = float(df['resolution'].max())
locked_resolution_scaled = (LOCKED_RESOLUTION_REAL - scaler_mean[idx_resolution]) / scaler_scale[idx_resolution]
print(f"🔒 Locked resolution (highest precision) = {LOCKED_RESOLUTION_REAL:.1f}")
print(f"✅ period is optimized; t_g is fixed to {FIXED_T_G:.3f} (not included in training)")

# Initialize optimizable parameters (start from the mean of training data for easier convergence)
optimizable_params = torch.randn(1, input_dim, requires_grad=True, device=device)
# Use the mean of training data as the initial value (in standardized space)
with torch.no_grad():
    mean_scaled = torch.mean(torch.tensor(X_train, dtype=torch.float32), dim=0).to(device)
    optimizable_params.data[0] = mean_scaled

# Only lock resolution (period participates in optimization)
optimizable_params.data[0, idx_resolution] = locked_resolution_scaled

# Base learning rate for inverse optimization (a new optimizer is created for each restart)
BASE_INVERSE_LR = 0.006

# EPSILON is defined in the data preprocessing step (computed dynamically based on DECIMAL_PLACES)
# Use a global variable here to ensure consistency

def enforce_constraints_rings(params_real_tensor, idx_n_rings, idx_r1, idx_t_r, idx_t_s, idx_period, epsilon, decimal_places):
    """Enforce/fix Rings parameters to satisfy constraints (within the specified precision; exclude t_g)"""
    with torch.no_grad():
        multiplier = 10 ** decimal_places
        params_rounded = torch.round(params_real_tensor * multiplier) / multiplier
        
        n_rings = int(torch.clamp(torch.round(params_rounded[0, idx_n_rings]), 1, 10).item())
        period = params_rounded[0, idx_period]
        size_limit = period / 2 - 0.001
        
        # Extract radii and ensure r_1 >= r_2 >= ... >= r_n, and r_i > 0
        radii = [max(params_rounded[0, idx_r1 + j].item(), 0.001) for j in range(10)]
        for j in range(n_rings):
            radii[j] = max(radii[j], 0.001)
        for j in range(n_rings, 10):
            radii[j] = 0.0
        
        # Sort the first n_rings radii from large to small
        active_radii = sorted(radii[:n_rings], reverse=True)
        for j in range(n_rings):
            params_rounded[0, idx_r1 + j] = min(active_radii[j], size_limit)  # r_i < period/2
        for j in range(n_rings, 10):
            params_rounded[0, idx_r1 + j] = 0.0
        
        params_rounded[0, idx_n_rings] = n_rings
        
        # Ensure t_r, t_s > 0 (t_g is fixed and not optimized)
        params_rounded[0, idx_t_r] = max(params_rounded[0, idx_t_r].item(), 0.001)
        params_rounded[0, idx_t_s] = max(params_rounded[0, idx_t_s].item(), 0.001)
        
        return params_rounded

# ==========================================
# Frequency band and optimization target configuration
# ==========================================
WL_RANGE_MIN = 0.3   # Full-band minimum value (μm)
WL_RANGE_MAX = 2.0   # Full-band maximum value (μm)

# auto: automatically find the widest continuous band with >90% absorption across the full range
# fixed: optimize within the specified band (can reuse 0.6-1.4 μm)
OPTIMIZATION_MODE = 'auto'  # 'auto' or 'fixed'
TARGET_WL_MIN = 0.6
TARGET_WL_MAX = 1.4

# Absorption thresholds
ABSORPTION_THRESHOLD_HIGH = 0.95  # High absorption threshold (95%)
ABSORPTION_THRESHOLD_GOOD = 0.90  # Good absorption threshold (90%)

# Steepness for the differentiable threshold approximation (larger -> closer to a hard threshold)
SOFT_THRESHOLD_STEEPNESS = 35.0
SOFT_MIN_STEEPNESS = 25.0
USE_DATASET_BEST_SEED = True


def longest_contiguous_band(mask, wavelengths):
    """Find the longest contiguous True band in a boolean mask (returns bandwidth, range, and point count)"""
    true_idx = np.where(mask)[0]
    if len(true_idx) == 0:
        return 0.0, [None, None], 0

    groups = np.split(true_idx, np.where(np.diff(true_idx) != 1)[0] + 1)
    longest = max(groups, key=len)
    start_idx = int(longest[0])
    end_idx = int(longest[-1])
    bandwidth = float(wavelengths[end_idx] - wavelengths[start_idx]) if len(longest) > 1 else 0.0
    wl_range = [float(wavelengths[start_idx]), float(wavelengths[end_idx])]
    return bandwidth, wl_range, len(longest)


def compute_band_metrics_np(spectrum_segment, wavelengths_segment, high_threshold, good_threshold):
    """Compute absorption metrics for a specified frequency band"""
    high_mask = spectrum_segment >= high_threshold
    good_mask = spectrum_segment >= good_threshold
    peak95_mask = spectrum_segment >= 0.95
    peak99_mask = spectrum_segment >= 0.99

    high_bandwidth, high_wl_range, high_points = longest_contiguous_band(high_mask, wavelengths_segment)
    good_bandwidth, good_wl_range, good_points = longest_contiguous_band(good_mask, wavelengths_segment)

    if np.any(high_mask):
        mean_above_high = float(np.mean(spectrum_segment[high_mask]))
    else:
        mean_above_high = 0.0

    top_k = max(1, int(np.ceil(len(spectrum_segment) * 0.10)))
    top10_mean = float(np.mean(np.sort(spectrum_segment)[-top_k:]))

    return {
        'avg_abs': float(np.mean(spectrum_segment)),
        'min_abs': float(np.min(spectrum_segment)),
        'max_abs': float(np.max(spectrum_segment)),
        'high_ratio': float(np.mean(high_mask)),
        'good_ratio': float(np.mean(good_mask)),
        'peak95_ratio': float(np.mean(peak95_mask)),
        'peak99_ratio': float(np.mean(peak99_mask)),
        'top10_mean': top10_mean,
        'mean_above_high': mean_above_high,
        'high_longest_bandwidth': high_bandwidth,
        'good_longest_bandwidth': good_bandwidth,
        'high_longest_range': high_wl_range,
        'good_longest_range': good_wl_range,
        'high_point_count': int(high_points),
        'good_point_count': int(good_points),
    }


def compute_composite_score(metrics_dict, wl_span):
    """Composite score: favor both broadband and peaks (allow dips)"""
    high_longest_ratio = metrics_dict['high_longest_bandwidth'] / max(wl_span, 1e-8)
    good_longest_ratio = metrics_dict['good_longest_bandwidth'] / max(wl_span, 1e-8)
    return (
        # Broadband part (65%)
        high_longest_ratio * 0.30 +
        good_longest_ratio * 0.15 +
        metrics_dict['high_ratio'] * 0.12 +
        metrics_dict['good_ratio'] * 0.08 +
        # Peak part (35%)
        metrics_dict['peak95_ratio'] * 0.10 +
        metrics_dict['top10_mean'] * 0.10 +
        metrics_dict['max_abs'] * 0.07 +
        metrics_dict['mean_above_high'] * 0.08
    )


def metrics_sort_key(metrics_dict):
    """Hard-metric sorting: broadband first, then peaks"""
    return (
        metrics_dict['high_longest_bandwidth'],
        metrics_dict['good_longest_bandwidth'],
        metrics_dict['high_ratio'],
        metrics_dict['good_ratio'],
        metrics_dict['peak95_ratio'],
        metrics_dict['top10_mean'],
        metrics_dict['max_abs'],
    )


wavelengths_full = np.linspace(WL_RANGE_MIN, WL_RANGE_MAX, output_dim)

if OPTIMIZATION_MODE == 'auto':
    target_idx_start = 0
    target_idx_end = output_dim
    optimization_label = f"Auto Full Range ({WL_RANGE_MIN:.1f}-{WL_RANGE_MAX:.1f} μm)"
elif OPTIMIZATION_MODE == 'fixed':
    target_idx_start = int(np.argmin(np.abs(wavelengths_full - TARGET_WL_MIN)))
    target_idx_end = int(np.argmin(np.abs(wavelengths_full - TARGET_WL_MAX))) + 1
    optimization_label = f"Fixed Band ({TARGET_WL_MIN:.1f}-{TARGET_WL_MAX:.1f} μm)"
else:
    raise ValueError(f"Unknown OPTIMIZATION_MODE: {OPTIMIZATION_MODE}")

target_wavelengths = wavelengths_full[target_idx_start:target_idx_end]
target_wl_span = float(target_wavelengths[-1] - target_wavelengths[0]) if len(target_wavelengths) > 1 else 0.0

print(f"\n--- Band optimization configuration ---")
print(f"Full-band range: {WL_RANGE_MIN:.1f} - {WL_RANGE_MAX:.1f} μm ({output_dim} points)")
print(f"Optimization mode: {optimization_label}")
print(f"Optimization interval indices: {target_idx_start} - {target_idx_end - 1}")
print(f"Optimization interval actual wavelengths: {target_wavelengths[0]:.3f} - {target_wavelengths[-1]:.3f} μm")
print(f"Optimization objective: automatically maximize the longest continuous bandwidth (Abs >= {ABSORPTION_THRESHOLD_HIGH:.2f})")
print("Strict constraints (Rings): n_rings 1~10; r_1 >= r_2 >= ... >= r_n; r_i < period/2; period optimized; t_r, t_s > 0; t_g fixed")

# Scan the dataset and build a seed pool: rank by hard metrics first, then do multi-start restarts
NUM_TOP_SEEDS = 15        # Increase seed count; start from better samples in the dataset
NUM_PERTURB_RESTARTS = 10  # Increase restart count
STEPS_PER_RESTART = 900   # More steps per restart
EVAL_EVERY = 50
INIT_NOISE_STD = 0.05
PENALTY_WEIGHT = 2000.0 # Physical-constraint penalty weight: large => hard constraints (may sacrifice spectrum performance); small => softer constraints (may sacrifice physics)

# Broadband optimization reward weights (emphasize >90% continuous bandwidth and coverage)
REWARD_HIGH_COVERAGE = 22.0      # >90% coverage (increase): the higher the fraction of Abs > 90% in the spectrum, the better
REWARD_HIGH_CONTINUITY = 20.0    # >90% continuous bandwidth (increase; core metric): the longer the Abs > 90% continuous band, the better
REWARD_GOOD_COVERAGE = 14.0      # >80% coverage: the higher the fraction of Abs > 80%, the better
REWARD_GOOD_CONTINUITY = 10.0    # >80% continuous bandwidth: the longer the Abs > 80% continuous band, the better
REWARD_PEAK95_COVERAGE = 10.0    # >95% peak coverage: the higher the fraction of Abs > 99%, the better
REWARD_PEAK99_COVERAGE = 5.0     # >99% peak coverage: the higher the fraction of Abs > 99%, the better
REWARD_PEAK95_CONTINUITY = 5.0   # >95% peak continuity: the longer the continuous band with Abs > 95%, the better
REWARD_SOFT_PEAK_MEAN = 8.0      # Weighted mean peak absorption: weighted average absorption in the high-absorption band
REWARD_TARGET_AVG = 5.0          # Target-range average absorption: average absorption within the optimization interval
REWARD_FULL_AVG = 0.5            # Full-band average (slightly increased to balance long wavelengths): average absorption over 0.3–2.0 μm
PEAK_SOFTMAX_TEMP = 18.0         # Softmax temperature: increase for more focus on the high-absorption band; decrease to make weights more uniform (less only about peaks)

dataset_ranked = []
for row_idx in range(len(y)):
    row_spectrum = y[row_idx, target_idx_start:target_idx_end]
    row_metrics = compute_band_metrics_np(
        row_spectrum,
        target_wavelengths,
        ABSORPTION_THRESHOLD_HIGH,
        ABSORPTION_THRESHOLD_GOOD
    )
    row_score = compute_composite_score(row_metrics, target_wl_span)
    dataset_ranked.append((row_idx, row_metrics, row_score))

dataset_ranked.sort(
    key=lambda item: (*metrics_sort_key(item[1]), item[2]),
    reverse=True
)
top_seed_indices = [item[0] for item in dataset_ranked[:NUM_TOP_SEEDS]]

dataset_best_idx = None
dataset_best_score = -float('inf')
dataset_best_metrics = None
if dataset_ranked:
    dataset_best_idx, dataset_best_metrics, dataset_best_score = dataset_ranked[0]

if dataset_best_metrics is not None:
    print("\n📎 Best dataset sample (ranked by hard metrics):")
    print(f"   - Longest BW(>90%): {dataset_best_metrics['high_longest_bandwidth']:.3f} μm")
    print(f"   - Coverage(>90%): {dataset_best_metrics['high_ratio']:.2%}")
    print(f"   - Peak Coverage(>95%): {dataset_best_metrics['peak95_ratio']:.2%}")
    print(f"   - Top10% Mean Abs: {dataset_best_metrics['top10_mean']:.3f}")
    print(f"   - Mean(Abs|>90%): {dataset_best_metrics['mean_above_high']:.3f}")
    print(f"   - Avg Abs: {dataset_best_metrics['avg_abs']:.3f}")
    if top_seed_indices:
        print(f"   - Top seeds (idx): {top_seed_indices}")

mean_scaled_seed = torch.mean(torch.tensor(X_train, dtype=torch.float32), dim=0).to(device)


def build_initial_scaled(seed_idx=None, noise_std=0.0):
    """Build the initial point in standardized space and lock resolution."""
    if seed_idx is not None:
        init = torch.tensor(X_scaled[seed_idx], dtype=torch.float32, device=device)
    else:
        init = mean_scaled_seed.clone()
    if noise_std > 0:
        init = init + torch.randn_like(init) * noise_std
    init[idx_resolution] = locked_resolution_scaled
    return init


restart_plans = []
if USE_DATASET_BEST_SEED and top_seed_indices:
    for seed_idx in top_seed_indices:
        restart_plans.append(("seed", seed_idx, 0.0))
for i in range(NUM_PERTURB_RESTARTS):
    perturb_seed = top_seed_indices[i % len(top_seed_indices)] if top_seed_indices else None
    restart_plans.append(("perturb", perturb_seed, INIT_NOISE_STD))
restart_plans.append(("mean", None, 0.0))

best_score = -float('inf')
best_key = (-float('inf'),) * 7
best_params = None
best_metrics = None

print(
    f"\n🚀 Start multi-start reverse optimization: {len(restart_plans)} restarts, "
    f"{STEPS_PER_RESTART} steps/restart, lr={BASE_INVERSE_LR}"
)
print(
    "🎯 Broadband + Peak Reward Weight: "
    f"HighCov={REWARD_HIGH_COVERAGE}, HighCont={REWARD_HIGH_CONTINUITY}, "
    f"GoodCov={REWARD_GOOD_COVERAGE}, GoodCont={REWARD_GOOD_CONTINUITY}, "
    f"Peak95={REWARD_PEAK95_COVERAGE}, Peak99={REWARD_PEAK99_COVERAGE}, "
    f"PeakMean={REWARD_SOFT_PEAK_MEAN}, TargetAvg={REWARD_TARGET_AVG}"
)

for restart_id, (mode, seed_idx, noise_std) in enumerate(restart_plans, start=1):
    init_scaled = build_initial_scaled(seed_idx=seed_idx, noise_std=noise_std)
    optimizable_params = init_scaled.view(1, -1).clone().detach().requires_grad_(True)
    opt_optimizer = optim.AdamW([optimizable_params], lr=BASE_INVERSE_LR, weight_decay=1e-6)

    restart_best_key = (-float('inf'),) * 7
    restart_best_score = -float('inf')
    restart_label = f"{mode}:{seed_idx}" if seed_idx is not None else mode
    print(f"\n--- Restart {restart_id}/{len(restart_plans)} [{restart_label}] ---")

    for step in range(STEPS_PER_RESTART):
        opt_optimizer.zero_grad()

        # Only lock resolution (period participates in optimization)
        optimizable_params.data[0, idx_resolution] = locked_resolution_scaled

        predicted_spectrum = model(optimizable_params)
        params_real = optimizable_params * scaler_scale + scaler_mean

        # Extract parameters (for physical constraints on Rings)
        p_period = params_real[0, idx_period]
        p_n_rings = int(torch.round(params_real[0, idx_n_rings]).clamp(1, 10).item())
        p_radii = [params_real[0, idx_r1 + j] for j in range(10)]

        spectrum = predicted_spectrum[0] if predicted_spectrum.dim() > 1 else predicted_spectrum
        target_spectrum = spectrum[target_idx_start:target_idx_end]

        # Optimize broadband + peaks together, without penalizing dips
        soft_high = torch.sigmoid((target_spectrum - ABSORPTION_THRESHOLD_HIGH) * SOFT_THRESHOLD_STEEPNESS)
        soft_good = torch.sigmoid((target_spectrum - ABSORPTION_THRESHOLD_GOOD) * SOFT_THRESHOLD_STEEPNESS)
        soft_peak95 = torch.sigmoid((target_spectrum - 0.95) * SOFT_THRESHOLD_STEEPNESS)
        soft_peak99 = torch.sigmoid((target_spectrum - 0.99) * SOFT_THRESHOLD_STEEPNESS)
        peak_weights = torch.softmax(target_spectrum * PEAK_SOFTMAX_TEMP, dim=0)

        soft_high_ratio = torch.mean(soft_high)
        soft_good_ratio = torch.mean(soft_good)
        soft_peak95_ratio = torch.mean(soft_peak95)
        soft_peak99_ratio = torch.mean(soft_peak99)
        soft_peak_mean = torch.sum(peak_weights * target_spectrum)
        target_avg = torch.mean(target_spectrum)
        full_avg = torch.mean(spectrum)

        if target_spectrum.numel() > 1:
            continuity_high = torch.mean(soft_high[:-1] * soft_high[1:])
            continuity_good = torch.mean(soft_good[:-1] * soft_good[1:])
            continuity_peak95 = torch.mean(soft_peak95[:-1] * soft_peak95[1:])
        else:
            continuity_high = torch.tensor(0.0, device=device)
            continuity_good = torch.tensor(0.0, device=device)
            continuity_peak95 = torch.tensor(0.0, device=device)

        broadband_objective = (
            soft_high_ratio * REWARD_HIGH_COVERAGE +       # Encourage >90% broadband
            continuity_high * REWARD_HIGH_CONTINUITY +     # >90% continuous bandwidth
            soft_good_ratio * REWARD_GOOD_COVERAGE +       # Encourage >80% broadband
            continuity_good * REWARD_GOOD_CONTINUITY +     # >80% continuous bandwidth
            soft_peak95_ratio * REWARD_PEAK95_COVERAGE +  # Peak coverage >95%
            soft_peak99_ratio * REWARD_PEAK99_COVERAGE +  # Peak coverage >99%
            continuity_peak95 * REWARD_PEAK95_CONTINUITY +  # Peak continuity
            soft_peak_mean * REWARD_SOFT_PEAK_MEAN +       # Weighted mean peak absorption
            target_avg * REWARD_TARGET_AVG +               # Target-range average absorption
            full_avg * REWARD_FULL_AVG                     # Secondary reference
        )
        main_loss = -broadband_objective

        # Strict physical constraint penalty (Rings)
        penalty = 0.0
        penalty += torch.sum(torch.relu(0.001 - params_real))
        size_limit = p_period / 2 - 0.001
        for j in range(p_n_rings):
            penalty += torch.relu(p_radii[j] - size_limit)  # r_i < period/2
        for j in range(p_n_rings - 1):
            penalty += torch.relu(p_radii[j + 1] - p_radii[j])  # r_i >= r_{i+1}

        total_loss = main_loss + penalty * PENALTY_WEIGHT
        total_loss.backward()

        # Locked parameter is not updated
        if optimizable_params.grad is not None:
            optimizable_params.grad[0, idx_resolution] = 0.0

        opt_optimizer.step()

        # Project back into the feasible region after the update
        optimizable_params.data[0, idx_resolution] = locked_resolution_scaled
        params_real_current = optimizable_params * scaler_scale + scaler_mean
        params_real_corrected = enforce_constraints_rings(
            params_real_current, idx_n_rings, idx_r1, idx_t_r, idx_t_s, idx_period, EPSILON, DECIMAL_PLACES
        )
        optimizable_params.data[0] = (params_real_corrected[0] - scaler_mean) / scaler_scale
        optimizable_params.data[0, idx_resolution] = locked_resolution_scaled

        should_eval = (step % EVAL_EVERY == 0) or (step == STEPS_PER_RESTART - 1)
        if not should_eval:
            continue

        with torch.no_grad():
            corrected_spectrum = model(optimizable_params)
            corrected_spectrum_1d = corrected_spectrum[0] if corrected_spectrum.dim() > 1 else corrected_spectrum
            corrected_spectrum_np = corrected_spectrum_1d.cpu().numpy()

            target_spectrum_np = corrected_spectrum_np[target_idx_start:target_idx_end]
            range_metrics = compute_band_metrics_np(
                target_spectrum_np,
                target_wavelengths,
                ABSORPTION_THRESHOLD_HIGH,
                ABSORPTION_THRESHOLD_GOOD
            )
            current_key = metrics_sort_key(range_metrics)
            score = compute_composite_score(range_metrics, target_wl_span)

            if current_key > restart_best_key:
                restart_best_key = current_key
                restart_best_score = score

            if (current_key > best_key) or (current_key == best_key and score > best_score):
                best_key = current_key
                best_score = score
                best_params = params_real_corrected.clone().detach()
                best_metrics = {
                    'score': score,
                    'range_metrics': range_metrics,
                    'full_avg_abs': float(np.mean(corrected_spectrum_np)),
                    'spectrum': corrected_spectrum_np.copy()
                }

            if step % (EVAL_EVERY * 2) == 0 or step == STEPS_PER_RESTART - 1:
                print(
                    f"R{restart_id} S{step:4d} | "
                    f"BW90={range_metrics['high_longest_bandwidth']:.3f} μm | "
                    f"BW80={range_metrics['good_longest_bandwidth']:.3f} μm | "
                    f"Cov90={range_metrics['high_ratio']:.2%} | "
                    f"Cov80={range_metrics['good_ratio']:.2%} | "
                    f"Peak95={range_metrics['peak95_ratio']:.2%} | "
                    f"Top10={range_metrics['top10_mean']:.3f} | "
                    f"Mean>90={range_metrics['mean_above_high']:.3f} | "
                    f"Avg={range_metrics['avg_abs']:.3f} | Penalty={penalty.item():.5f}"
                )

    print(
        f"Restart {restart_id} done | Best BW90={restart_best_key[0]:.3f} μm, "
        f"BW80={restart_best_key[1]:.3f} μm, "
        f"Cov90={restart_best_key[2]:.2%}, Cov80={restart_best_key[3]:.2%}, "
        f"Peak95={restart_best_key[4]:.2%}, Top10={restart_best_key[5]:.3f}, PeakMax={restart_best_key[6]:.3f}"
    )

# Final result (use the best parameters and ensure constraints are satisfied)
if best_params is not None:
    best_params_real = best_params[0].cpu().numpy()
    final_spectrum = best_metrics['spectrum']
    final_range_metrics = best_metrics['range_metrics']
    full_avg_abs = best_metrics['full_avg_abs']
else:
    optimizable_params.data[0, idx_resolution] = locked_resolution_scaled
    params_real_final = optimizable_params * scaler_scale + scaler_mean
    best_params_corrected = enforce_constraints_rings(
        params_real_final, idx_n_rings, idx_r1, idx_t_r, idx_t_s, idx_period, EPSILON, DECIMAL_PLACES
    )
    best_params_real = best_params_corrected[0].cpu().numpy()

    final_params_tensor = torch.tensor(best_params_real.reshape(1, -1), dtype=torch.float32).to(device)
    final_params_scaled_normalized = (final_params_tensor - scaler_mean) / scaler_scale
    final_spectrum = model(final_params_scaled_normalized).detach().cpu().numpy()[0]
    final_range_metrics = compute_band_metrics_np(
        final_spectrum[target_idx_start:target_idx_end],
        target_wavelengths,
        ABSORPTION_THRESHOLD_HIGH,
        ABSORPTION_THRESHOLD_GOOD
    )
    full_avg_abs = float(np.mean(final_spectrum))
    best_score = compute_composite_score(final_range_metrics, target_wl_span)

target_avg_abs = final_range_metrics['avg_abs']
target_min_abs = final_range_metrics['min_abs']
target_max_abs = final_range_metrics['max_abs']
high_abs_ratio = final_range_metrics['high_ratio']
good_abs_ratio = final_range_metrics['good_ratio']
peak95_ratio = final_range_metrics['peak95_ratio']
peak99_ratio = final_range_metrics['peak99_ratio']
top10_mean_abs = final_range_metrics['top10_mean']
high_abs_bandwidth = final_range_metrics['high_longest_bandwidth']
good_abs_bandwidth = final_range_metrics['good_longest_bandwidth']
high_abs_wl_range = final_range_metrics['high_longest_range']
good_abs_wl_range = final_range_metrics['good_longest_range']

print("\n" + "=" * 70)
print(f"🎉 Optimization complete! {optimization_label} performance metrics")
print("=" * 70)
print(f"📊 Performance in optimization range:")
print(f"   - Average absorption rate: {target_avg_abs * 100:.2f}%")
print(f"   - Minimum absorption rate: {target_min_abs * 100:.2f}%")
print(f"   - Maximum absorption rate: {target_max_abs * 100:.2f}%")
print(f"   - Longest continuous bandwidth (>90%): {high_abs_bandwidth:.3f} μm")
if high_abs_wl_range[0] is not None:
    print(f"     Wavelength range: {high_abs_wl_range[0]:.3f} - {high_abs_wl_range[1]:.3f} μm")
print(f"   - Longest continuous bandwidth (>80%): {good_abs_bandwidth:.3f} μm")
if good_abs_wl_range[0] is not None:
    print(f"     Wavelength range: {good_abs_wl_range[0]:.3f} - {good_abs_wl_range[1]:.3f} μm")
print(f"   - Coverage (>90%): {high_abs_ratio * 100:.1f}%")
print(f"   - Coverage (>80%): {good_abs_ratio * 100:.1f}%")
print(f"   - Peak coverage (>95%): {peak95_ratio * 100:.1f}%")
print(f"   - Peak coverage (>99%): {peak99_ratio * 100:.1f}%")
print(f"   - Top 10% mean absorption: {top10_mean_abs * 100:.2f}%")
print(f"\n📊 Full band (0.3-2.0 μm) mean absorption: {full_avg_abs * 100:.2f}%")
print(f"📊 Composite score: {best_score:.4f}")
if dataset_best_metrics is not None:
    print(f"📎 Best dataset sample Longest BW(>90%): {dataset_best_metrics['high_longest_bandwidth']:.3f} μm")
print("=" * 70)
print("🎯 Predicted best Rings parameters")
print("=" * 70)
p = dict(zip(feature_cols, best_params_real))
p_rounded = {k: round(v, DECIMAL_PLACES) for k, v in p.items()}

# Output the optimal number of radii and their values
best_n_rings = int(round(p_rounded['n_rings']))
best_radii = [p_rounded[f'r_{i+1}'] for i in range(best_n_rings)]
print(f"\n📐 Optimal number of Rings (n_rings): {best_n_rings}")
# Compact format for easy copy
print(f"\nn_rings={best_n_rings}")
print(f"radius={','.join(f'{r:.{DECIMAL_PLACES}f}' for r in best_radii)}")
print(f"t_r={p_rounded['t_r']:.{DECIMAL_PLACES}f}")
print(f"t_s={p_rounded['t_s']:.{DECIMAL_PLACES}f}")
print(f"t_g={FIXED_T_G:.{DECIMAL_PLACES}f}")
print(f"period={p_rounded['period']:.{DECIMAL_PLACES}f}")
print("=" * 70)
print(f"📌 Note: resolution locked to {LOCKED_RESOLUTION_REAL:.1f} (highest precision)")

# Constraint check (specified precision) for Rings
print(f"\n--- Constraint check ({DECIMAL_PLACES} decimal places, Rings) ---")
print(
    f"🔒 Resolution locked: {p['resolution']:.1f} (target: {LOCKED_RESOLUTION_REAL:.1f}) "
    f"{'✅' if abs(p['resolution'] - LOCKED_RESOLUTION_REAL) < 1e-5 else '❌'}"
)
print(f"✅ n_rings: {best_n_rings} (1~10)")
print(f"✅ Used radii r_1~r_{best_n_rings}: {'Decreasing ✅' if all(best_radii[i] >= best_radii[i+1] for i in range(len(best_radii)-1)) else '❌'}")
size_limit = round(p_rounded['period'] / 2, DECIMAL_PLACES)
print(f"✅ All r_i < period/2 ({size_limit:.{DECIMAL_PLACES}f}): {all(r < size_limit for r in best_radii)}")
print(f"✅ t_r, t_s > 0: {p_rounded['t_r'] > 0 and p_rounded['t_s'] > 0}; t_g fixed={FIXED_T_G:.{DECIMAL_PLACES}f}")

# Save the training curves plot
plt.tight_layout()
training_plot_filename = f'/Users/xiebailin/meep_projects/FYP/fyp_semB/prediction/training_history_ring_{LR_SCHEDULER_TYPE}.png'
plt.savefig(training_plot_filename, dpi=150)
print(f"\n📊 Training curves plot saved to: training_history_ring_{LR_SCHEDULER_TYPE}.png")
plt.show()

# Plot the optimization result
plt.figure(figsize=(12, 7))
wavelengths = np.linspace(WL_RANGE_MIN, WL_RANGE_MAX, output_dim)
plt.plot(wavelengths, final_spectrum, label='Optimized Spectrum', color='green', linewidth=2.5, zorder=3)

if OPTIMIZATION_MODE == 'fixed':
    plt.axvspan(TARGET_WL_MIN, TARGET_WL_MAX, alpha=0.12, color='yellow', label='Optimization Range', zorder=1)
    target_wavelengths_plot = wavelengths[target_idx_start:target_idx_end]
    target_spectrum_plot = final_spectrum[target_idx_start:target_idx_end]
    plt.plot(
        target_wavelengths_plot,
        target_spectrum_plot,
        label=f'Target Band ({TARGET_WL_MIN:.1f}-{TARGET_WL_MAX:.1f} μm)',
        color='red',
        linewidth=2.8,
        zorder=4,
        alpha=0.85
    )

if high_abs_wl_range[0] is not None:
    plt.axvspan(
        high_abs_wl_range[0],
        high_abs_wl_range[1],
        alpha=0.18,
        color='red',
        label='Longest Continuous >90% Band',
        zorder=2
    )

plt.axhline(
    y=ABSORPTION_THRESHOLD_HIGH,
    color='orange',
    linestyle='--',
    linewidth=1.5,
    label=f'High Threshold ({ABSORPTION_THRESHOLD_HIGH * 100:.0f}%)',
    alpha=0.7,
    zorder=2
)
plt.axhline(
    y=ABSORPTION_THRESHOLD_GOOD,
    color='blue',
    linestyle='--',
    linewidth=1.5,
    label=f'Good Threshold ({ABSORPTION_THRESHOLD_GOOD * 100:.0f}%)',
    alpha=0.7,
    zorder=2
)

title_text = (
    f"Optimized Absorption Spectrum ({optimization_label})\n"
    f"Longest BW(>90%): {high_abs_bandwidth:.3f} μm, "
    f"Coverage(>90%): {high_abs_ratio:.1%}, Avg/Min: {target_avg_abs:.3f}/{target_min_abs:.3f}"
)
plt.title(title_text, fontsize=11)
plt.xlabel("Wavelength (μm)", fontsize=12)
plt.ylabel("Absorption", fontsize=12)
plt.ylim(0, 1)
plt.xlim(WL_RANGE_MIN, WL_RANGE_MAX)
plt.grid(True, alpha=0.3, zorder=0)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
output_filename = f'/Users/xiebailin/meep_projects/FYP/fyp_semB/prediction/optimization_result_ring_{LR_SCHEDULER_TYPE}.png'
plt.savefig(output_filename, dpi=150)
print(f"📊 Optimization result plot saved to: optimization_result_ring_{LR_SCHEDULER_TYPE}.png")
plt.show()

# ==========================================
# 6. Save the predicted absorption curve to a CSV file
# ==========================================
print("\n--- Saving predicted absorption curve to CSV file ---")

# Prepare data: wavelength, transmission, reflection, and absorption
# Physical relation: A + T + R = 1
# For an ideal absorber, assume transmission T ≈ 0, so reflection R = 1 - A
absorption = final_spectrum
transmission = np.zeros_like(absorption)  # Assume transmission is 0 (ideal absorber)
reflection = 1.0 - absorption  

# Ensure reflection is non-negative
reflection = np.maximum(reflection, 0.0)

# Create a DataFrame
spectrum_df = pd.DataFrame({
    'Wavelength(um)': wavelengths,
    'Transmission(T)': transmission,
    'Reflection(R)': reflection,
    'Absorption(A)': absorption
})

# Use a fixed filename and overwrite the old file every time
csv_filename = f'/Users/xiebailin/meep_projects/FYP/fyp_semB/prediction/predicted_spectrum_ring.csv'

# Save the CSV file (using scientific notation; consistent with the example file)
spectrum_df.to_csv(csv_filename, index=False, float_format='%.6e')
print(f"✅ Predicted absorption curve saved to: {csv_filename}")
print(f"   - Wavelength range: {WL_RANGE_MIN:.3f} - {WL_RANGE_MAX:.3f} μm")
print(f"   - Number of data points: {len(wavelengths)}")
print(f"   - Mean absorption: {np.mean(absorption):.6f}")
print(f"   - Max absorption: {np.max(absorption):.6f}")
print(f"   - Min absorption: {np.min(absorption):.6f}")
