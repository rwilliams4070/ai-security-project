import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * gradient.sign()
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    return adversarial_image

criterion = nn.CrossEntropyLoss()

def evaluate_clean(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Clean Accuracy: {accuracy * 100:.2f}%")
    return accuracy * 100

def evaluate_fgsm(model, epsilon):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        gradient = images.grad.data
        adversarial_images = fgsm_attack(images, epsilon, gradient)

        adversarial_outputs = model(adversarial_images)
        predictions = adversarial_outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"FGSM Accuracy at epsilon={epsilon}: {accuracy * 100:.2f}%")
    return accuracy * 100

def adversarial_train(model, epochs=2, epsilon=0.20):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            gradient = images.grad.data
            adversarial_images = fgsm_attack(images, epsilon, gradient)

            optimizer.zero_grad()
            defended_outputs = model(adversarial_images.detach())
            defended_loss = criterion(defended_outputs, labels)

            defended_loss.backward()
            optimizer.step()

            total_loss += defended_loss.item()

        print(f"Defense Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/baseline_model.pt", map_location=device))

    print("\nBefore Defense:")
    before_clean = evaluate_clean(model)
    before_attack = evaluate_fgsm(model, epsilon=0.20)

    print("\nStarting adversarial training defense...")
    adversarial_train(model, epochs=2, epsilon=0.20)

    print("\nAfter Defense:")
    after_clean = evaluate_clean(model)
    after_attack = evaluate_fgsm(model, epsilon=0.20)

    torch.save(model.state_dict(), "models/defended_model.pt")
    print("Defended model saved to models/defended_model.pt")

    labels = ["Clean Before", "Attack Before", "Clean After", "Attack After"]
    values = [before_clean, before_attack, after_clean, after_attack]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Performance Before and After Defense")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("graphs/defense_comparison.png")

    print("Saved graph to graphs/defense_comparison.png")
