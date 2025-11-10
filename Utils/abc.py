from Utils.model_methods import PL_Model
import torch


class ABC(PL_Model):
    def __init__(self, backbone, classifier, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3},
                 lr_scheduler=None, lr_scheduler_params=dict(), attack=None,
                 positive_class=1, loss_fn=torch.nn.functional.binary_cross_entropy,
                 freeze_backbone=True, use_hidden_layer_of_backbone=True,
                 neg_labels=False, seed=42, device='cuda'):
        super(ABC, self).__init__(backbone, classifier, optimizer=optimizer, optimizer_params=optimizer_params,
                 lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params, attack=attack,
                 positive_class=positive_class, loss_fn=loss_fn,
                 freeze_backbone=freeze_backbone, use_hidden_layer_of_backbone=use_hidden_layer_of_backbone,
                 neg_labels=neg_labels, seed=seed, device=device)

    def forward(self, x):
        if self.feature_extractor is None:
            representations = x
        else:
            representations = self.feature_extractor(x).squeeze()
        if len(representations.shape) == 1:
            # batch size is 1
            representations = representations.unsqueeze(dim=0)
        #         print((representations).shape)
        #         print(self.classifier(representations).shape)
        recon_error = torch.linalg.norm(x - self.classifier(representations), ord=2, dim=-1)
        return torch.exp(-recon_error)
