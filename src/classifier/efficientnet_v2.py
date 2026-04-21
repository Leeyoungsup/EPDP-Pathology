"""EfficientNetV2-based subtype classifier (EffNet-Real / EffNet-Syn).

Backbone is loaded from ``timm`` and the final classifier head is replaced
with a single Linear layer matching the number of histopathologic subtypes.
The optimizer used in the manuscript is Sharpness-Aware Minimization (SAM)
wrapped around an SGD base optimizer.
"""
import timm
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Drop the original classifier head, keep the global-pooled features."""

    def __init__(self, str_backbone: str = "tf_efficientnetv2_l"):
        super().__init__()
        cnn = timm.create_model(str_backbone, pretrained=True)
        self.feature_ex = nn.Sequential(*list(cnn.children())[:-1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_ex(inputs)


class EfficientNetV2Classifier(nn.Module):
    def __init__(self, int_num_classes: int, int_image_feature_dim: int,
                 feature_extractor: FeatureExtractor):
        super().__init__()
        self.num_classes = int_num_classes
        self.image_feature_dim = int_image_feature_dim
        self.feature_extractor = feature_extractor
        self.classification_layer = nn.Linear(int_image_feature_dim, int_num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        features = features.view(features.size(0), -1)
        logits = self.classification_layer(features)
        return logits


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Wraps a base optimizer (e.g., SGD) and adds an inner ascent step
    that climbs the local loss surface before the descent step.
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure callable"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
