import torch
from torch import nn
import lightning as L
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAveragePrecision


def bump_activation(x, sigma=0.5):
    return torch.exp(-0.5 * torch.square(x) / torch.square(sigma))


class Bump(nn.Module):
    def __init__(self, sigma=0.5, trainable=False):
        super(Bump, self).__init__()
        self.sigma = sigma
        self.sigma_factor = nn.Parameter(
            torch.tensor(self.sigma, dtype=torch.float32), requires_grad=trainable)

    def forward(self, inputs):
        return bump_activation(inputs, self.sigma_factor)


class RBFLayer(nn.Module):
    def __init__(self, units, beta=1., initializer=nn.init.xavier_normal_, dim=1, seed=2023):
        super(RBFLayer, self).__init__()
        self.units = units
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.dim = dim
        if initializer is None:
            mu = torch.ones(units)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if dim == 1:
                mu = initializer((units,))
            else:
                mu = initializer(units)
        # if dim == 1:
        self.mu = nn.Parameter(mu, requires_grad=True)
        # else:
        #     self.mu = nn.Parameter(initializer(units), requires_grad=True)

    def forward(self, inputs):
        if self.dim == 1:
            inputs_expanded = torch.unsqueeze(inputs, 1)
            diff = inputs_expanded - self.mu
            l2 = torch.sum(torch.pow(diff, 2), dim=2)
        # elif self.dim == 2:
        else:
            x = inputs[:, :, None]
            mu = self.mu[None, None, :]
            diff = x - mu
            l2 = torch.sum(torch.pow(diff, 2), dim=-1)
        # elif self.dim == 4:
        #     # Reshape inputs to (batch_size, channels, height * width)
        #     inputs_reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
        #
        #     # Calculate L2 distance between inputs and centroids
        #     diff = inputs_reshaped[:, None, :, :] - self.centroids[None, :, :, :]
        #     l2 = torch.sum(torch.pow(diff, 2), dim=3)
        #     l2 = torch.sum(l2, dim=2)
        res = torch.exp(-1 / self.beta * l2)
        return res


# define the LightningModule
class PL_Model(L.LightningModule):
    def __init__(self, backbone, classifier, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3},
                 lr_scheduler=None, lr_scheduler_params=dict(), attack=None,
                 positive_class=1, loss_fn=torch.nn.functional.binary_cross_entropy,
                 freeze_backbone=True, use_hidden_layer_of_backbone=True,
                 neg_labels=False, seed=42, device='cuda'):
        """
        Supervised AD Model
        :param backbone: frozen backbone feature extractor
        :param classifier: classifier head
        :param optimizer: torch.optim object
        :param positive_class: label of positive class for binary classification (negative class is anomalies)
        :param loss_fn: loss function
        :param neg_labels: False for 0/1 label, True for -1/+1 label
        :param seed: random seed
        :param device: cpu or cuda
        """
        super().__init__()
        self.positive_class = positive_class
        self.attack = attack

        ############# FREEZE BACKBONE ###################
        if backbone is None:
            self.feature_extractor = None
        else:
            if use_hidden_layer_of_backbone:
                all_layers = list(backbone.children())
                # rep_dim = all_layers[-1][0].in_features
                layers = all_layers[:-1]
                self.feature_extractor = nn.Sequential(*layers)
            else:
                self.feature_extractor = backbone
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                self.feature_extractor.eval()
        ################################################

        L.seed_everything(seed, workers=True)
        self.classifier = classifier
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.loss_fn = loss_fn
        self.neg_labels = neg_labels

        #         self.model = nn.Sequential(self.feature_extractor, self.classifier)

        #         self.training_step_outputs = []
        #         self.validation_step_outputs = []
        #         self.test_step_outputs = []
        self.metric_names = ["precision", "recall", "f1", "average_precision"]
        # self.train_metrics = [BinaryPrecision().to(device), BinaryRecall().to(device),
        #                       BinaryF1Score().to(device), BinaryAveragePrecision().to(device)]
        # self.val_metrics = [BinaryPrecision().to(device), BinaryRecall().to(device),
        #                     BinaryF1Score().to(device), BinaryAveragePrecision().to(device)]
        self.test_metrics = [BinaryPrecision().to(device), BinaryRecall().to(device),
                             BinaryF1Score().to(device), BinaryAveragePrecision().to(device)]


    def forward(self, x):
        # with torch.no_grad():
        if self.feature_extractor is None:
            representations = x
        else:
            representations = self.feature_extractor(x).squeeze()
        if len(representations.shape) == 1:
            # batch size is 1
            representations = representations.unsqueeze(dim=0)
        #         print((representations).shape)
        #         print(self.classifier(representations).shape)
        return self.classifier(representations).squeeze(dim=-1)

    def get_metrics(self, y_pred, y, positive_class=None):
        #         x, y = batch
        #         y_pred = self.model(x)
        if positive_class is None:
            positive_class = self.positive_class
        if positive_class == 0:
            y = 1. - y
            y_pred = 1. - y_pred
        precision = BinaryPrecision()(y_pred, y)
        recall = BinaryRecall()(y_pred, y)
        f1 = BinaryF1Score()(y_pred, y)
        aupr = BinaryAveragePrecision()(y_pred, y)
        return precision, recall, f1, aupr

    def log_metrics(self, y_pred, y, prefix):
        for i, (metric_name, metric) in enumerate(zip(self.metric_names, self.test_metrics)):
            if i != 3:
                self.log(f"{prefix}_{metric_name}", metric(torch.sigmoid(y_pred), y.int()), prog_bar=True,
                         on_epoch=True)

    def get_loss(self, y_pred, y):
        if self.neg_labels:
            y = (2 * y - 1).float()
            y_pred = (2 * y_pred - 1).float()
        loss = self.loss_fn(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx, adv_training=False, model=None):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.forward(x)
        loss = self.get_loss(y_pred, y)
        # if self.neg_labels:
        #     y = 2 * y - 1
        #     y_pred = 2 * y_pred - 1
        # loss = self.loss_fn(y_pred, y)
        #         loss = self.get_loss(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        if self.positive_class == 0:
            y = 1. - y
            y_pred = 1. - y_pred
        self.log_metrics(y_pred, y, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred, y)
        #         loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        if self.positive_class == 0:
            y = 1. - y
            y_pred = 1. - y_pred
        self.log_metrics(y_pred, y, prefix="val")
        return loss

    #     def on_validation_epoch_end(self):
    #         y_pred = torch.stack(self.validation_step_outputs)
    #         metrics = self.get_metrics(batch, batch_idx)
    #         self.log("val metrics (P, R, F1, AUPR):", metrics)
    #         self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.forward(x)
        if self.positive_class == 0:
            y = 1. - y
            y_pred = 1. - y_pred
        self.log_metrics(y_pred, y, prefix="test")

    #         self.test_step_outputs.append(y_pred)

    #     def on_test_epoch_end(self):
    #         y_pred = torch.stack(self.validation_step_outputs)
    #         metrics = self.get_metrics(batch, batch_idx)
    #         self.log("test metrics (P, R, F1, AUPR):", metrics)
    #         self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        opt = self.optimizer(self.classifier.parameters(), **self.optimizer_params)
        if self.lr_scheduler is None:
            return opt
        sch = self.lr_scheduler(self.optimizer, **self.lr_scheduler_params)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1,}}
        # return [opt], [sch]


class DeepSVDD(L.LightningModule):
    def __init__(self, encoder, centre, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3},
                 lr_scheduler=None, lr_scheduler_params=dict(), attack=None, seed=42):
        """
        Supervised AD Model
        :param encoder: pre-trained encoder
        :param optimizer: torch.optim object
        :param seed: random seed
        :param device: cpu or cuda
        """
        super().__init__()
        self.attack = attack

        self.feature_extractor = encoder
        ################################################

        L.seed_everything(seed, workers=True)
        centre.requires_grad = True
        self.centre = centre
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.loss_fn = torch.nn.functional.mse_loss

    def forward(self, x):
        representations = self.feature_extractor(x)#.squeeze()
        dist = representations - self.centre
        if len(dist.shape) == 1:
            # batch size is 1
            neg_dist = -dist.norm().unsqueeze(0)
        else:
            # normal is more positive. will get squared in loss anyways
            neg_dist = -dist.norm(dim=1)
        # print(neg_dist.size())
        return neg_dist

    def get_loss(self, y_pred):
        loss = self.loss_fn(y_pred, torch.zeros_like(y_pred))
        return loss

    def training_step(self, batch, batch_idx, adv_training=False, model=None):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # add it here in case we need to do adversarial training later on
        if adv_training and model is not None:
            x = self.attack(model, x, y, epsilon=0.03, alpha=0.01, num_iter=40, random_start=True,
                                device=self.device)
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred)
        #         loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    #     def on_validation_epoch_end(self):
    #         y_pred = torch.stack(self.validation_step_outputs)
    #         metrics = self.get_metrics(batch, batch_idx)
    #         self.log("val metrics (P, R, F1, AUPR):", metrics)
    #         self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.forward(x)
        loss = self.get_loss(y_pred)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = self.optimizer(self.feature_extractor.parameters(), **self.optimizer_params)
        if self.lr_scheduler is None:
            return opt
        sch = self.lr_scheduler(self.optimizer, **self.lr_scheduler_params)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1,}}
        # return [opt], [sch]


def build_classifier(classifier_layers, rep_dim, activation=nn.LeakyReLU(), one_class=True, dropout=0,
                     sigmoid_head=True, seed=42):
    if seed is not None:
        L.seed_everything(seed, workers=True)
    classifier_head = nn.ModuleList()
    if type(rep_dim) is int:
        for i in range(classifier_layers-1):
            classifier_head.append(nn.Linear(rep_dim, rep_dim))
            classifier_head.append(activation)
            if dropout > 0:
                classifier_head.append(nn.Dropout(p=dropout))
        classifier_head.append(nn.Linear(rep_dim, rep_dim))
    else:
        for i in range(len(rep_dim)-2):
            classifier_head.append(nn.Linear(rep_dim[i], rep_dim[i+1]))
            classifier_head.append(activation)
            if dropout > 0:
                classifier_head.append(nn.Dropout(p=dropout))
        classifier_head.append(nn.Linear(rep_dim[-2], rep_dim[-1]))
        rep_dim = rep_dim[-1]

    if one_class:
        classifier_head.append(Bump(sigma=0.5))
        classifier_head.append(RBFLayer(units=1, initializer=None))
    else:
        classifier_head.append(activation)
        classifier_head.append(nn.Linear(rep_dim, 1))
        if sigmoid_head:
            classifier_head.append(nn.Sigmoid())

    classifier = nn.Sequential(*classifier_head)

    return classifier


def build_model(rep_dim, activation=nn.LeakyReLU(), bias=False, batchnorm=True, dropout=0, seed=42):
    if seed is not None:
        L.seed_everything(seed, workers=True)
    model = nn.ModuleList()
    for i in range(len(rep_dim)-2):
        model.append(nn.Linear(rep_dim[i], rep_dim[i+1], bias=bias))
        model.append(activation)
        if batchnorm:
            model.append(nn.BatchNorm1d(rep_dim[i+1], affine=False, eps=1e-10))
        if dropout > 0:
            model.append(nn.Dropout(p=dropout))
    model.append(nn.Linear(rep_dim[-2], rep_dim[-1], bias=bias))
    return nn.Sequential(*model)


def hinge_loss(y_pred, y_true):
    y_label = y_true.detach().clone()
    y_label[y_label == 0] = -1
    loss = 1. - y_pred * y_label
    loss[loss < 0] = 0.
    return torch.mean(loss)
