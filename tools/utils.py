import os
import wandb

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def setup_wandb(args):
    """Login to Weights & Biases and initialize a new run."""

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args),  # Include all args dynamically
    )
    return run