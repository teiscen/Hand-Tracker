import torch 
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),  # input: 6 features (example), compress to 4
            nn.ReLU(),
            nn.Linear(4, 2),  # bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 6),  # reconstruct to original size
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
X_smooth_tensor = torch.tensor(X_smooth, dtype=torch.float32)

for epoch in range(1000):
    output = model(X_raw_tensor)
    loss = criterion(output, X_smooth_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    corrected = model(torch.tensor(new_noisy_data, dtype=torch.float32))

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())  # Should print True if GPU is working

if __name__ == "__main__":
    main()