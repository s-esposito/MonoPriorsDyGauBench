import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(0)
x_full = torch.linspace(-10, 10, 100).reshape(-1, 1)
true_y = torch.sin(x_full) * 3 + x_full**2 / 5 - x_full  # More complex true function
dy = torch.randn_like(true_y) * 5  # Bias/noise term
y_biased = true_y + dy

# Select 10 sparse samples for biased data
sparse_indices = torch.tensor([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
x_sparse_biased = x_full[sparse_indices]
y_sparse_biased = y_biased[sparse_indices]

# Select 10 sparse samples for error-free data
x_sparse_clean = x_full[sparse_indices]
y_sparse_clean = true_y[sparse_indices]

# Define a simple model function
def train_model(x, y, epochs=1000):
    model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model

# Train models
model_biased = train_model(x_sparse_biased, y_sparse_biased)
model_clean = train_model(x_sparse_clean, y_sparse_clean)

# Predictions
predicted_y_biased = model_biased(x_full).detach()
predicted_y_clean = model_clean(x_full).detach()

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(x_sparse_biased, y_sparse_biased, label='Biased Data (y + dy)', color='blue', alpha=0.5, s=100)
plt.scatter(x_sparse_clean, y_sparse_clean, label='Error-free Data (y)', color='orange', alpha=0.5, s=100)
plt.plot(x_full, true_y, label='True Function', color='green', linewidth=2)
plt.plot(x_full, predicted_y_biased, label='Fitted Function (Biased)', color='red', linewidth=2, linestyle='dashed')
plt.plot(x_full, predicted_y_clean, label='Fitted Function (Clean)', color='purple', linewidth=2, linestyle='dotted')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Effect of Training with Biased and Error-free Sparse Labels')

# Save the figure to disk
plt.savefig('toy_posefix.png')

# Optional: Close the plot if you're running this in an environment where plots are displayed
plt.close()
