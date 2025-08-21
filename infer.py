import click
import torch
import numpy as np
from train import ResidualNoisePredictor, DiffusionModel, Evaluator
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class Infer:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.model = ResidualNoisePredictor().to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.diffusion = DiffusionModel(self.model, device=device)
        
    def generate_samples(self, num_samples, num_inference_steps=50, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        with torch.no_grad():
            samples = self.diffusion.sample(
                num_samples=num_samples, 
                num_inference_steps=num_inference_steps
            ).cpu()
        
        return samples
    
    def visualize_samples(self, samples, output_path="generated_samples.png", title="Generated Samples"):
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.7)
        plt.title(title)
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")


@click.command()
@click.option("--ckpt_path", default="model.pth", help="Path to the trained model checkpoint")
@click.option("--num_samples", default=1000, help="Number of samples to generate")
@click.option("--num_inference_steps", default=50, help="Number of inference steps for DDIM sampling")
@click.option("--seed", default=None, type=int, help="Random seed for reproducible generation")
@click.option("--output_path", default="generated_samples.png", help="Path to save the visualization")
@click.option("--evaluate", is_flag=True, help="Run evaluation metrics against ground truth")
@click.option("--device", default="cuda", help="Device to run inference on")
def main(ckpt_path, num_samples, num_inference_steps, seed, output_path, evaluate, device):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    infer = Infer(ckpt_path, device=device)
    
    print(f"Generating {num_samples} samples with {num_inference_steps} inference steps...")
    samples = infer.generate_samples(
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    infer.visualize_samples(samples, output_path)
    
    if evaluate:
        class DummyTrainer:
            def __init__(self, diffusion):
                self.diffusion = diffusion
                self.train_dataset = torch.tensor(
                    make_moons(n_samples=20_000, noise=0.1, random_state=42)[0],
                    dtype=torch.float32,
                )
        
        dummy_trainer = DummyTrainer(infer.diffusion)
        evaluator = Evaluator(dummy_trainer, num_samples=len(samples))
        
        # don't resample, use the existing ones we just generated!
        def mock_sample(*args, **kwargs):
            return samples.to(device)
        
        original_sample = infer.diffusion.sample
        infer.diffusion.sample = mock_sample
        
        evaluator.eval()
        
        infer.diffusion.sample = original_sample
    
    print(f"Inference complete! Generated {len(samples)} samples.")


if __name__ == "__main__":
    main()
