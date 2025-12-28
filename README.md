# RealNVP MNIST Generator 🎨

This project implements a Conditional RealNVP (Normalizing Flow) model to generate handwritten digits (MNIST). It is implemented using PyTorch and includes a web demo.

## 📁 Project Structure
- `network.py`: Contains the model architecture (RealNVP, Coupling Layers).
- `main.py`: Script for training the model. It integrates **WandB** for real-time loss tracking and image logging.
- `app.py`: Interactive web demo using **Gradio**.
- `models/`: Directory to save/load trained model weights (`.pth`).
- `requirements.txt`: List of dependencies.

## 🚀 How to run

### 1. Installation
    Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
### 2. Training (Optional)
    If you want to train the model from scratch, you can change the value of hyperparameters in main.py: (Note: You will need a WandB account)
    ```bash
    wandb login
    python main.py

### 3. Running the demo
    Launch the Gradio web interface to generate images:
    ```bash
    python app.py
    

