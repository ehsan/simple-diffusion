Two Moons Diffusion
===================

This repo implements a tiny diffusion model that learns to generate samples from the two moons dataset.  The entire work appears in a Jupyter Notebook (`project.ipynb`) for easy following and reproducing the result.

## Modeling the problem as diffusion
We formulate the problem as follows: given the coordinates of samples from a two moons distribution, in the forward pass we gradually add noise to the data until it becomes indistinguishable from Gaussian noise (this provides a simple starting distribution for the reverse process).  Then we train a model that learns the reverse of this process by predicting the noise added across time.  We follow DDPM for training the model because it's an easy to implement and stable formulation for diffusion, but for sampling we follow DDIM which is a deterministic version of the Euler method (given `eta=0.0`).

## Evaluation metrics
We use both a visual comparison as well as three numeric metrics in order to assess the quality of the model that we train.  The metrics are as follows.  We combine these metrics with equal weight to calculate an overall metric that we aim to minimize.

### Distance metric
This metric is based on the distance of the generated samples to the real samples.  We use the kNN algorithm to find the nearest sample to each of the generated samples.  We select a cut-off radius of 0.02 and calculate how many of the generated samples fall within that radius of the real samples.  We calculate this as (1 - proportion_within_radius) so that lower values indicate better performance (more generated samples are close to real samples).  This helps make the metric consistent with MSE.

### MSE metric
This metric is based on the MSE between the generated and real samples.  We can't use an MSE directly because the order of the samples matters for MSE, so instead we convert the real and generated samples into a normalized histogram and calculate the MSE between the histograms.  We are careful to construct the histograms based on the range of both the generated and real samples to ensure that the histogram comparison is fair (e.g., it would punish models that generate wild outliers).

### Shape metric
One thing that these metrics don't quite measure yet is the visual shape; we want the generated samples to form two separated clusters.  The clustering itself is straightforward; we can use kMeans clustering and ensure we get two clusters.  In order to ensure the clusters are separate, we use the silhouette score as a metric (there may be better ways of measuring that, but this seems to get the job done).

## Improving the baseline model
We use the `NoisePredictor` model as a baseline, trained for 1000 epochs at a batch size of 512.  This network is extremely simple:

```python
class NoisePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = torch.cat([x, t_embed], dim=-1)
        return self.net(x)
```

Here are the metrics from the baseline model:
```
Distance: 0.5920
MSE: 0.0096
Shape: 0.2635
Overall: 0.2884
```

And here are the visual results.

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/d701e250-3dfe-4bff-bef4-d7ecffbd3b2b" />

Once the baseline is established, we run a series of tests with different network architectures.

We first try to increase the network size, which seems to improve results up to a certain extent and then asymptotes.

Next we try to add residual connections to the network (`ResidualNoisePredictor`), which improves results a lot, both evident from visual comparison and the decrease in metrics (improving over the baseline by about 9.14%). Here are the visual results:

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/628850f0-df22-4b60-be08-fc5e84426127" />

We also try using sinusoidal time embedding with `ResidualNoisePredictor` which is also unfortunately not helpful.

Next we try a mini-UNet architecture, but unfortunately it is not an improvement over `ResidualNoisePredictor`, neither visually nor based on the metrics, so we pick `ResidualNoisePredictor` as the winning architecture choice:

```python
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
    def __init__(self, input_dim=2, hidden_dim=512, time_embed_dim=64, num_res_blocks=5, num_layers_time_embed=2):
        super().__init__()
        
        time_embed_layers = [
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers_time_embed - 1):
            time_embed_layers.extend([
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.ReLU(),
            ])

        self.time_embed = nn.Sequential(*time_embed_layers)
        
        self.input_proj = nn.Linear(input_dim + time_embed_dim, hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        x = torch.cat([x, t_embed], dim=-1)
        
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_proj(x)
```

Next we try data augmentation through 3 types of transforms on the dataset (rotation, scaling and translation).  Unfortunately this kind of data augmentation significantly degrades performance, increasing our combined loss metric by 31.19% and degrading visual results significantly (indicating worse sample quality).

<img width="980" height="528" alt="image" src="https://github.com/user-attachments/assets/4e7bf0e3-e450-49df-8a86-363f532668bd" />

## Training for longer
We also show that the simple `NoisePredictor` network, if trained for longer (e.g., 5000 epochs) is still capable of learning the two moons distribution quite well.  We ran an ablation against training the `ResidualNoisePredictor` network for the same number of epochs and the visual results are very similar (and from the metrics perspective, the results from `NoisePredictor` actually beat the results from `ResidualNoisePredictor`).  In other words, adding residual connections helps make the training more compute efficient, but probably, given the relatively simple nature of the dataset, a much simpler network is still capable of learning it very well.

You can see the visual results from `NoisePredictor` and `ResidualNoisePredictor` respectively below, both trained for 5000 epochs.
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/3c27a175-e443-483d-8f35-213991007044" />
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/00d56817-1e30-4cb8-860b-6aa1bad7588c" />

## Ideas for future improvements
Besides the ideas we tried in this small project, there are many other potential ideas for improving on top of our baseline that we can try in the future; here are some examples:
* Trying different learning rates or different learning rate schedules (e.g., cosine annealing schedule)
* Trying different optimizer parameters (e.g., weight decay)
* Trying gradient clipping
* Trying other activation functions, e.g., SiLU or GELU
* Trying non-linear noise schedules (e.g., cosine)
* Trying to sample with more DDIM steps
* Trying smaller or more moderate data augmentations
* Trying loss weighting based on the timestep