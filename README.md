# AI Security Project

## Project Overview
This project demonstrates how machine learning systems can be attacked and defended using adversarial machine learning techniques. A neural network was trained on the MNIST handwritten digit dataset using PyTorch. The Fast Gradient Sign Method (FGSM) attack was implemented to fool the model by slightly modifying image pixels. A defense strategy using adversarial training was then applied to improve robustness against attacks.

---

## Tools Used
- Python 3
- PyTorch
- torchvision
- NumPy
- Matplotlib
- GitHub
- Kali Linux Virtual Machine

---

## Baseline Model Results
- Dataset: MNIST
- Model Type: Convolutional Neural Network (CNN)
- Baseline Accuracy: 98.70%

The model successfully classified handwritten digits with high accuracy before any attacks were introduced.

---

## FGSM Attack Results
The Fast Gradient Sign Method (FGSM) attack was used to create adversarial examples.

### Accuracy Under Attack
| Epsilon | Accuracy |
|----------|-----------|
| 0.00 | 98.70% |
| 0.05 | 95.58% |
| 0.10 | 86.55% |
| 0.15 | 64.60% |
| 0.20 | 34.44% |
| 0.25 | 10.44% |
| 0.30 | 3.09% |

The attack successfully reduced model accuracy as perturbation strength increased.

---

## Defense Results
Adversarial training was implemented as a defense technique.

### Before Defense
- Clean Accuracy: 98.70%
- FGSM Accuracy (epsilon=0.2): 34.44%

### After Defense
- Clean Accuracy: 96.20%
- FGSM Accuracy (epsilon=0.2): 97.83%

The defended model became significantly more resistant to adversarial attacks.

---

## How to Run

### Train Baseline Model
```bash
python3 baseline_model.py
