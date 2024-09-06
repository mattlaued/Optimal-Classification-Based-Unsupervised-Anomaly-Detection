from Utils.pipeline import *

dataset_name = "kdd"
if dataset_name == "kitsune":
    kwargs_data = {"mirai": False, "custom": "fuzzing"}
    n_dim = 115
elif dataset_name == "kdd":
    kwargs_data = {}
    n_dim = 119
else:
    raise ValueError("Dataset name must be either 'kitsune' or 'kdd'")

val_split = 0.2

# only select normal class for training
training_classes = [0]
scaler = MinMaxScaler()

train_index_match_col = 'attack_map'
train_label_col = 'normal_flag'
test_index_match_col = 'attack_map'
test_label_col = 'attack_map'

model_type = "NN"  # NN, SVM, IsolationForest, DeepSVDD
one_class = True
sigmoid_head = True
classifier_layers = 2 + 1
n_neurons = 500
rep_dim = [n_dim] + [n_neurons for i in range(classifier_layers - 1)]
assert len(rep_dim) == classifier_layers
activation = torch.nn.LeakyReLU()
dropout = 0
# epochs = 2
epochs = 200
# weight_decay = 0
weight_decay = 1e-2
optimizer_params = {'lr': 1e-3}
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# lr_scheduler = LinearWarmupCosineAnnealingLR
# lr_scheduler_params = {"warmup_epochs": epochs//10, "max_epochs": epochs}
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
# lr_scheduler_params = {"T_max": epochs}
lr_scheduler = None
lr_scheduler_params = dict()
use_hinge = False
patience = 7
batch_size = 1024
repeats = 3

theory_nn = True
custom = True

synthetic_anom_ratio = 1
synthetic_val_anom_constant = True
synthetic_anom_test_ratio = False
one_hot_col_len = None
binary_cols = True
delta = 0.

fpr = [0.05, None]
eval_only = False

args = {
    "repeats": repeats,
    "model_type": model_type,
    "classifier_layers": classifier_layers,
    "rep_dim": rep_dim,
    "one_class": one_class,
    "sigmoid_head": sigmoid_head,
    "activation": activation,
    "dropout": dropout,
    "epochs": epochs,
    "weight_decay": weight_decay,
    "optimizer_params": optimizer_params,
    "lr_scheduler": lr_scheduler,
    "lr_scheduler_params": lr_scheduler_params,
    "use_hinge": use_hinge,
    "patience": patience,
    "batch_size": batch_size,
    "theory_nn": theory_nn,
    "custom": custom,
    "num_real_training": 1,   # can change the proportion of real normal data used during training
    "synthetic_anom_ratio": synthetic_anom_ratio,
    "synthetic_anom_test_ratio": synthetic_anom_test_ratio,
    "synthetic_val_anom_constant": synthetic_val_anom_constant,
    "one_hot_col_len": one_hot_col_len,
    "binary_cols": binary_cols,
    "delta": delta,
    "fpr": fpr,
    "eval_only": eval_only
}

if __name__ == "__main__":
    aggregated_results = main(
        dataset_name, kwargs_data, val_split, training_classes, scaler,
        train_index_match_col, train_label_col, test_index_match_col, test_label_col,
        **args
    )
