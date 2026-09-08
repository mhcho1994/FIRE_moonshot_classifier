"""Inference branch of DiversifyFlight, without optimizers or training imports."""
import torch
from torch import nn

from fire_moonshot_classifier.networks import FlightFeaturizer, FeatBottleneck, PrototypeClassifier


class DiversifyNetwork(nn.Module):
    def __init__(self, n_feat, cnn_ch, bottleneck_dim, attn_hidden):
        super().__init__()
        self.featurizer = FlightFeaturizer(n_feat, cnn_ch, attn_hidden)
        self.bottleneck = FeatBottleneck(cnn_ch, bottleneck_dim)
        self.classifier = PrototypeClassifier(2, bottleneck_dim)

    def forward(self, x):
        f1, _, h = self.featurizer.forward_features(x)
        z = self.bottleneck(h)
        return self.classifier(z), f1.mean(dim=-1)


def inference_state(state):
    """Select the inference branch and verify all of its checkpoint keys."""
    try:
        model_config = {
            "n_feat": int(state["featurizer.block1.0.weight"].shape[1]),
            "cnn_ch": int(state["featurizer.block3.0.weight"].shape[0]),
            "bottleneck_dim": int(state["bottleneck.fc.weight"].shape[0]),
            "attn_hidden": int(state["featurizer.attn_pool.attention.0.weight"].shape[0]),
        }
        weights = {
            k: v.detach().cpu().clone() for k, v in state.items()
            if k.startswith(("featurizer.", "bottleneck.", "classifier."))
        }
        model = DiversifyNetwork(**model_config)
        model.load_state_dict(weights, strict=True)
    except (KeyError, AttributeError, IndexError, TypeError, RuntimeError) as exc:
        raise ValueError("Unsupported checkpoint: expected the CNN + prototype Diversify model") from exc
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Model weights contain non-finite values")
    return model_config, weights


def knn_distances(features, bank, k, bank_batch_size=4096):
    """Exact L1 kNN distance with bounded query-by-reference memory."""
    features = features.detach().cpu().float()
    best = torch.full((len(features), k), float("inf"))
    for chunk in bank.split(bank_batch_size):
        distances = torch.cdist(features, chunk)
        best = torch.cat((best, distances), dim=1).topk(k, largest=False, dim=1).values
    return best[:, -1]
