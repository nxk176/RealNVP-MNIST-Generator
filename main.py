import wandb
import argparse
import network

class Config:
    def __init__(self):
        self.data_dir = "./data"
        self.batch_size = 128
        self.epochs = 100
        self.lr = 1e-3
        self.n_coupling = 8
        self.hidden_dim = 1024 
        self.sample_dir = "wandb_samples"
        self.no_cuda = False

if __name__ == "__main__":
    args = Config()
    wandb.init(
        project="RealNVP_Conditional_FCNN",
        # name="Robust_FCNN_MNIST",
        config=args
    )
    print("---Start training---")
    trained_model = network.train(args)
    wandb.finish()