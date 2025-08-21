import click
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm.auto import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class ResidualNoisePredictor(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=512,
        time_embed_dim=64,
        num_res_blocks=5,
        num_layers_time_embed=2,
    ):
        super().__init__()

        time_embed_layers = [
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers_time_embed - 1):
            time_embed_layers.extend(
                [
                    nn.Linear(time_embed_dim, time_embed_dim),
                    nn.ReLU(),
                ]
            )

        self.time_embed = nn.Sequential(*time_embed_layers)

        self.input_proj = nn.Linear(input_dim + time_embed_dim, hidden_dim)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.output_proj = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = torch.cat([x, t_embed], dim=-1)

        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_proj(x)


class DiffusionModel:
    def __init__(
        self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.model = model.to(device)

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def ddim_sample_step(self, x_t, t, t_prev):
        predicted_noise = self.model(x_t, t.float().unsqueeze(-1))

        alpha_cumprod_t = self.alpha_cumprod[t].reshape(-1, 1)
        alpha_cumprod_t_prev = (
            self.alpha_cumprod[t_prev].reshape(-1, 1)
            if t_prev >= 0
            else torch.ones_like(alpha_cumprod_t)
        )

        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        pred_x0 = (
            x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise
        ) / sqrt_alpha_cumprod_t

        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1.0 - alpha_cumprod_t_prev)

        dir_xt = sqrt_one_minus_alpha_cumprod_t_prev * predicted_noise

        x_prev = sqrt_alpha_cumprod_t_prev * pred_x0 + dir_xt

        return x_prev

    def sample(self, num_samples, num_inference_steps=50, shape=(2,)):
        self.model.eval()

        step_size = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size).long()
        timesteps = torch.cat([timesteps, torch.tensor([self.num_timesteps - 1])])
        timesteps = torch.flip(timesteps, [0])

        x = torch.randn(num_samples, *shape).to(self.device)

        for i, t in enumerate(tqdm(timesteps[:-1], desc="Sampling")):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 2 else -1

            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(self.device)

            with torch.no_grad():
                x = self.ddim_sample_step(x, t_batch, t_prev)

        return x

    def q_sample(self, x0, t, noise=None):
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(
            -1, 1
        )

        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

    def train_step(self, x0):
        batch_size = x0.shape[0]

        t = torch.randint(0, self.num_timesteps, (batch_size,)).to(self.device)
        noise = torch.randn_like(x0)

        x_t, _ = self.q_sample(x0, t, noise)

        predicted_noise = self.model(x_t, t.float().unsqueeze(-1))

        loss = F.mse_loss(predicted_noise, noise)

        return loss


class Trainer:
    def __init__(
        self,
        model,
        device="cuda",
        num_epochs=10,
        batch_size=64,
        num_batches_per_epoch=10,
        learning_rate=1e-4,
        num_train_samples=20_000,
    ):
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.num_train_samples = num_train_samples

        self.diffusion = DiffusionModel(model, device=device)
        self.optimizer = torch.optim.Adam(
            self.diffusion.model.parameters(), lr=learning_rate
        )

        self.train_dataset = torch.tensor(
            make_moons(n_samples=num_train_samples, noise=0.1, random_state=42)[0],
            dtype=torch.float32,
        )

    def prepare_batch(self):
        indices = torch.randperm(len(self.train_dataset))[: self.batch_size]
        batch = self.train_dataset[indices].to(self.device)

        return batch

    def train(self):
        self.diffusion.model.train()

        losses = []

        for _ in tqdm(range(self.num_epochs), desc="Training"):
            for _ in range(self.num_batches_per_epoch):
                batch = self.prepare_batch()

                self.optimizer.zero_grad()

                loss = self.diffusion.train_step(batch)
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())

        return losses

    def save_loss_curve(self, losses, file_name="loss_curve.png"):
        plt.plot(losses)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.grid()
        plt.savefig(file_name)

    def save_checkpoint(self, ckpt_path):
        torch.save(self.diffusion.model.state_dict(), ckpt_path)


class Evaluator:
    def __init__(self, trainer, num_samples=100):
        self.trainer = trainer
        self.diffusion = trainer.diffusion

        self.num_samples = num_samples

    def eval(self):
        with torch.no_grad():
            samples = self.diffusion.sample(self.num_samples).cpu()

        self._visualize(samples)

        metrics = self.calculate_metrics(samples)
        print(f"Distance: {metrics['distance']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"Shape: {metrics['shape']:.4f}")

        print(
            f"Overall: {(metrics['distance'] + metrics['mse'] + metrics['shape']) / 3.0:.4f}"
        )

    def calculate_metrics(self, generated_samples):
        real_samples = (
            self.trainer.train_dataset[: len(generated_samples)].clone().cpu().numpy()
        )
        generated_samples = generated_samples.numpy()

        return {
            "distance": self.calculate_distance_metric(generated_samples, real_samples),
            "mse": self.calculate_mse_metric(generated_samples, real_samples),
            "shape": self.calculate_shape_metric(generated_samples, real_samples),
        }

    def calculate_shape_metric(self, generated_samples, real_samples):
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(generated_samples)

        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            return 1.0

        silhouette = silhouette_score(generated_samples, labels)
        return (1.0 - silhouette) / 2.0

    def calculate_mse_metric(self, generated_samples, real_samples):
        all_samples = np.vstack([real_samples, generated_samples])
        x_min, x_max = all_samples[:, 0].min() - 0.5, all_samples[:, 0].max() + 0.5
        y_min, y_max = all_samples[:, 1].min() - 0.5, all_samples[:, 1].max() + 0.5
        bounds = [[x_min, x_max], [y_min, y_max]]

        real_hist, _, _ = np.histogram2d(
            real_samples[:, 0], real_samples[:, 1], bins=64, range=bounds
        )
        gen_hist, _, _ = np.histogram2d(
            generated_samples[:, 0], generated_samples[:, 1], bins=64, range=bounds
        )

        real_hist = real_hist / real_hist.max() if real_hist.max() > 0 else real_hist
        gen_hist = gen_hist / gen_hist.max() if gen_hist.max() > 0 else gen_hist

        return np.mean((real_hist - gen_hist) ** 2)

    def calculate_distance_metric(self, generated_samples, real_samples, radius=0.02):
        nn = NearestNeighbors(n_neighbors=1).fit(real_samples)
        distances, _ = nn.kneighbors(generated_samples)

        return 1.0 - (distances.flatten() < radius).astype(int).sum() / len(
            generated_samples
        )

    def _visualize(self, samples, file_name="evaluation.png"):
        gt = self.trainer.train_dataset.clone().cpu()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Generated Samples")
        plt.scatter(samples[:, 0], samples[:, 1], s=5)
        plt.axis("equal")

        plt.subplot(1, 2, 2)
        plt.title("Ground Truth")
        plt.scatter(gt[:, 0], gt[:, 1], s=5)
        plt.axis("equal")

        plt.savefig(file_name)


@click.command()
@click.option("--random_seed", default=42, help="Random seed for reproducibility")
@click.option("--num_epochs", default=5_000, help="Number of training epochs")
@click.option("--batch_size", default=512, help="Batch size for training")
@click.option(
    "--num_batches_per_epoch", default=100, help="Number of batches per epoch"
)
@click.option("--lr", default=1e-4, help="Learning rate for training")
@click.option("--ckpt_path", default="model.pth", help="Path to save the model checkpoint")
def main(random_seed, num_epochs, batch_size, num_batches_per_epoch, lr, ckpt_path):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_default_device("cuda")

    trainer = Trainer(
        model=ResidualNoisePredictor(),
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        learning_rate=lr,
    )
    losses = trainer.train()
    trainer.save_loss_curve(losses)

    ev = Evaluator(trainer, num_samples=1000)
    ev.eval()

    trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
