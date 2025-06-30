import os
import wandb
import argparse

def parse_args_vae():
    parser = argparse.ArgumentParser(description="Train VAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/vae/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Beta KL Loss") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--architecture", type=str, default="cuda", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED", help="Dataset path")
    parser.add_argument("--wandb_project", type=str, default="vae_training_exp1", help="Wandb Project Name")
    parser.add_argument("--wandb_entity", type=str, default="imagine-laboratory-conare", help="Wandb Project Entity")

    return parser.parse_args()

def parse_args_vae_hierarchical():
    parser = argparse.ArgumentParser(description="Train VAE Hierarchical on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/vae/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Beta KL Loss") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dataset", type=str, default="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED", help="Dataset path")
    parser.add_argument("--wandb_project", type=str, default="vae_training_exp1", help="Wandb Project Name")
    parser.add_argument("--wandb_entity", type=str, default="imagine-laboratory-conare", help="Wandb Project Entity")
    parser.add_argument("--loss", type=str, default="mse", help="Loss use for reconstruction: Perceptual Loss | MSE Loss")

    return parser.parse_args()

def parse_args_slot_vae():
    parser = argparse.ArgumentParser(description="Train Slot VAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/vae/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Beta KL Loss") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--architecture", type=str, default="cuda", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED", help="Dataset path")
    parser.add_argument("--wandb_project", type=str, default="vae_training_exp1", help="Wandb Project Name")
    parser.add_argument("--wandb_entity", type=str, default="imagine-laboratory-conare", help="Wandb Project Entity")

    return parser.parse_args()

def parse_args_vqvae():
    parser = argparse.ArgumentParser(description="Train VQVAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/vae/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Beta KL Loss") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--architecture", type=str, default="cuda", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED", help="Dataset path")
    parser.add_argument("--wandb_project", type=str, default="vae_training_exp1", help="Wandb Project Name")
    parser.add_argument("--wandb_entity", type=str, default="imagine-laboratory-conare", help="Wandb Project Entity")

    return parser.parse_args()

def parse_args_vqvae_hierarchical():
    parser = argparse.ArgumentParser(description="Train VQVAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/vae/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Beta KL Loss") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--architecture", type=str, default="cuda", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED", help="Dataset path")
    parser.add_argument("--wandb_project", type=str, default="vae_training_exp1", help="Wandb Project Name")
    parser.add_argument("--wandb_entity", type=str, default="imagine-laboratory-conare", help="Wandb Project Entity")

    return parser.parse_args()


def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def setup_wandb_vae(args):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "architecture": args.architecture,
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "beta_kl_loss": args.beta_kl_loss,
            "reconstruction_loss": "mse_pixel"
        },
    )
    return run


def setup_wandb_vqvae(args):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "architecture": args.architecture,
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "beta_kl_loss": args.beta_kl_loss,
            "reconstruction_loss": "mse_pixel"
        },
    )
    return run