import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dimensions, dropout=0.0, final_activation=False):
    layers = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        is_last = index == len(dimensions) - 2
        if not is_last or final_activation:
            # LayerNorm has no client-specific running statistics and is stable for
            # the small final batches common in federated partitions.
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ViewAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout):
        super().__init__()
        encoder_dims = [input_dim, *hidden_dims, embedding_dim]
        decoder_dims = [embedding_dim, *reversed(hidden_dims), input_dim]
        self.encoder = _mlp(encoder_dims, dropout=dropout)
        self.decoder = _mlp(decoder_dims, dropout=dropout)

    def encode(self, view):
        return self.encoder(view)

    def decode(self, embedding):
        return self.decoder(embedding)


class ClusteringHead(nn.Module):
    """DEC-style trainable cluster centers returning Student-t assignments."""
    def __init__(self, num_clusters, embedding_dim, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.centers = nn.Parameter(torch.empty(num_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embedding):
        distance = torch.sum((embedding.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        assignments = (1.0 + distance / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)


class MultiViewClusteringModel(nn.Module):
    def __init__(self, view_dims, num_clusters, hidden_dims, embedding_dim, dropout=0.0, alpha=1.0):
        super().__init__()
        self.view_models = nn.ModuleList(
            ViewAutoencoder(dim, hidden_dims, embedding_dim, dropout) for dim in view_dims
        )
        self.view_logits = nn.Parameter(torch.zeros(len(view_dims)))
        self.cluster_head = ClusteringHead(num_clusters, embedding_dim, alpha=alpha)

    def encode(self, views):
        view_embeddings = [model.encode(view) for model, view in zip(self.view_models, views)]
        normalized = [F.normalize(embedding, dim=1) for embedding in view_embeddings]
        weights = torch.softmax(self.view_logits, dim=0)
        fused = sum(weight * embedding for weight, embedding in zip(weights, normalized))
        return view_embeddings, F.normalize(fused, dim=1)

    def forward(self, views):
        view_embeddings, fused = self.encode(views)
        reconstructions = [
            model.decode(embedding) for model, embedding in zip(self.view_models, view_embeddings)
        ]
        assignments = self.cluster_head(fused)
        return {
            "view_embeddings": view_embeddings,
            "embedding": fused,
            "reconstructions": reconstructions,
            "assignments": assignments,
        }


def target_distribution(assignments):
    frequency = assignments.sum(dim=0).clamp_min(1e-12)
    weight = assignments.pow(2) / frequency
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)


def clustering_objective(
    outputs,
    views,
    loss_weights,
    clustering_enabled,
    clustering_weight_scale=1.0,
    target_assignments=None,
):
    reconstruction = torch.stack([
        F.mse_loss(reconstructed, original)
        for reconstructed, original in zip(outputs["reconstructions"], views)
    ]).mean()

    fused = outputs["embedding"]
    consistency = torch.stack([
        F.mse_loss(F.normalize(embedding, dim=1), fused)
        for embedding in outputs["view_embeddings"]
    ]).mean()

    assignments = outputs["assignments"]
    if clustering_enabled:
        target = (
            target_distribution(assignments.detach())
            if target_assignments is None
            else target_assignments.detach()
        )
        if target.shape != assignments.shape:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match "
                f"assignment shape {tuple(assignments.shape)}"
            )
        clustering = F.kl_div(assignments.clamp_min(1e-12).log(), target, reduction="batchmean")
        mean_assignment = assignments.mean(dim=0)
        balance = torch.sum(mean_assignment * torch.log(mean_assignment * assignments.shape[1] + 1e-12))
    else:
        clustering = assignments.new_zeros(())
        balance = assignments.new_zeros(())

    clustering_weight_scale = float(clustering_weight_scale) if clustering_enabled else 0.0
    total = (
        loss_weights["reconstruction"] * reconstruction
        + loss_weights["consistency"] * consistency
        + clustering_weight_scale * loss_weights["clustering"] * clustering
        + clustering_weight_scale * loss_weights["balance"] * balance
    )
    return total, {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "consistency": float(consistency.detach()),
        "clustering": float(clustering.detach()),
        "balance": float(balance.detach()),
        "confidence": float(assignments.max(dim=1).values.mean().detach()),
        "clustering_weight_scale": clustering_weight_scale,
    }
