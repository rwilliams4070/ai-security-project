import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

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

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/baseline_model.pt", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * gradient.sign()
    adversarial_image = image + perturbation
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    return adversarial_image

def evaluate_fgsm(epsilon):
    correct = 0
    total = 0
    saved_example = False

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

        if not saved_example:
            original_prediction = outputs.argmax(dim=1)[0].item()
            adversarial_prediction = predictions[0].item()
            true_label = labels[0].item()

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.title(f"Original\nTrue: {true_label}, Pred: {original_prediction}")
            plt.imshow(images[0].detach().cpu().squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title(f"Adversarial\nTrue: {true_label}, Pred: {adversarial_prediction}")
            plt.imshow(adversarial_images[0].detach().cpu().squeeze(), cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"images/fgsm_example_epsilon_{epsilon}.png")
            print(f"Saved adversarial example image to images/fgsm_example_epsilon_{epsilon}.png")
            saved_example = True

    accuracy = correct / total
    print(f"FGSM Accuracy with epsilon={epsilon}: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    print(f"Using device: {device}")

    epsilons = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accuracies = []

    for epsilon in epsilons:
        accuracy = evaluate_fgsm(epsilon)
        accuracies.append(accuracy * 100)

    plt.figure()
    plt.plot(epsilons, accuracies, marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("FGSM Attack: Accuracy Drop as Epsilon Increases")
    plt.grid(True)
    plt.savefig("graphs/fgsm_accuracy_drop.png")

    print("Saved graph to graphs/fgsm_accuracy_drop.png")
