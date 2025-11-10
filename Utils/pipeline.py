import numpy as np

from Utils.data_methods_synthetic import get_dataloader
from Utils.model_methods import *
from Utils.eval_methods import *
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor as LOF
from Utils.data_methods import *
from Utils.post_processing import *
from collections import OrderedDict
import shutil


def main(dataset_name, kwargs_data, val_split, training_classes, scaler,
         train_index_match_col, train_label_col, test_index_match_col, test_label_col,
         num_real_training, synthetic_anom_ratio, synthetic_val_anom_constant, one_hot_col_len, binary_cols, delta,
         model_type, classifier_layers, rep_dim, activation, one_class, dropout, sigmoid_head, theory_nn, custom,
         repeats, eval_only,
         epochs, weight_decay, optimizer_params, lr_scheduler, lr_scheduler_params, patience, use_hinge,
         batch_size,
         fpr,
         synthetic_anom_test_ratio=False,
         shallow_params={}, deepsvdd_ae_ckpt_path=None, deepsvdd_ERM=False, deepsvdd_params=None,
         nn_params=None, quadratic_bump=False, drocc_params=None,
         save_models=True
         ):
    (df, test_df, features_to_encode, numeric_features, normal_label, attack_labels, attacks, test_classes,
     raw_label_col, map_attack) = get_data(dataset_name, **kwargs_data)

    exp_name = dataset_name
    if "custom" in kwargs_data:
        exp_name += f"_{kwargs_data['custom']}"
    if "save_suffix" in kwargs_data:
        exp_name += f"_{kwargs_data['save_suffix']}"

    class_label = OrderedDict()
    for lab, num in zip(attack_labels, test_classes):
        class_label[lab] = num
    print("class_label", class_label)

    x_training_real, y_training_real, x_test, y_test, one_hot_col_length = preprocess(
        df, test_df, features_to_encode, numeric_features, training_classes=training_classes,
        map_attack=map_attack, raw_label_col=raw_label_col, normal_label=normal_label, scaler=scaler,
        train_index_match_col=train_index_match_col, train_label_col=train_label_col,
        test_index_match_col=test_index_match_col, test_label_col=test_label_col)
    one_hot_col_len = one_hot_col_length if one_hot_col_len is None else one_hot_col_len
    # n_dim = x_training_real.shape[-1]
    x_test, y_test = get_in_range_data(x_test, y_test, print_drop_count=True)
    print("x_training_real", len(x_training_real))
    print("x_test", len(x_test))
    if synthetic_anom_test_ratio:
        x_test, y_test = add_synthetic_anom_to_test_data(
            x_test, y_test, synthetic_anom_test_ratio, one_hot_col_length, seed_anom_generation=15973)
        class_label["Synthetic"] = len(class_label)

    # Baselines
    # label 0 for anom, 1 for normal
    # num_real_training can be less than 1, but proportions remain the same (on average), so no need to account for that
    num_normal = np.sum(y_training_real)
    num_att_total = np.sum(y_training_real == 0) + int(synthetic_anom_ratio * num_normal)
    # num_normal = len(y_training) - num_att_total
    print("Baseline train AUPR: ", 1 - num_normal / (num_normal + num_att_total))

    # label 0 for normal, 1/2/3/... for different attacks
    num_att_total = np.sum(y_test != 0)
    num_normal = len(y_test) - num_att_total
    print("Baseline overall AUPR: ", 1 - num_normal / len(y_test))

    val_counts = test_df['attack_map'].value_counts()

    for att in attacks:
        if att in val_counts:
            num_att = val_counts[att]
            print(f"Baseline AUPR {att}: ", num_att / (num_normal + num_att))
        else:
            print(f"Anomaly {att} is not in testing data.")
    if synthetic_anom_test_ratio:
        print(f"Baseline AUPR Synthetic: ", synthetic_anom_test_ratio / (1 + synthetic_anom_test_ratio))
    # #####################################################################

    # if __name__ == '__main__':
    args = {
        "repeats": repeats,
        "model_type": model_type,
        "classifier_layers": classifier_layers,
        "rep_dim": rep_dim,
        "one_class": one_class,
        "sigmoid_head": sigmoid_head,
        "quadratic_bump": quadratic_bump,
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
        # "optimizer": optimizer,
        # "loss_fn": loss_fn,
        # "neg_labels": neg_labels,
        "synthetic_anom_ratio": synthetic_anom_ratio,
        "synthetic_val_anom_constant": synthetic_val_anom_constant,
        "one_hot_col_len": one_hot_col_len,
        "binary_cols": binary_cols,
        "delta": delta,
        "fpr": fpr,
        "eval_only": eval_only
    }
    args = set_hyperparameters(custom=custom, hyperparam_dict=args)
    if type(fpr) is list:
        del args['fpr']
        aggregated_results = []
        agg, classifier_name = split_train_eval(x_training_real, y_training_real, num_real_training, x_test, y_test, val_split,
                               class_label, exp_name,
                               fpr=fpr[0], shallow_params=shallow_params, deepsvdd_ae_ckpt_path=deepsvdd_ae_ckpt_path,
                               deepsvdd_ERM=deepsvdd_ERM, deepsvdd_params=deepsvdd_params, nn_params=nn_params, drocc_params=drocc_params, **args
                               )
        aggregated_results.append(agg)
        print("*%***************")
        if not args['eval_only']:
            args['eval_only'] = True
        for q in fpr[1:]:
            agg, classifier_name = split_train_eval(
                x_training_real, y_training_real, num_real_training, x_test, y_test, val_split, class_label,
                exp_name,
                fpr=q, shallow_params=shallow_params, deepsvdd_ae_ckpt_path=deepsvdd_ae_ckpt_path,
                deepsvdd_ERM=deepsvdd_ERM, deepsvdd_params=deepsvdd_params, nn_params=nn_params, drocc_params=drocc_params, **args
            )
            aggregated_results.append(agg)
            print("*%***************")

    else:
        aggregated_results, classifier_name = split_train_eval(
            x_training_real, y_training_real, num_real_training, x_test, y_test, val_split, class_label, exp_name,
            shallow_params=shallow_params, deepsvdd_ae_ckpt_path=deepsvdd_ae_ckpt_path,
            deepsvdd_ERM=deepsvdd_ERM, deepsvdd_params=deepsvdd_params, nn_params=nn_params, drocc_params=drocc_params, **args
        )
    # delete saved model
    if not save_models:
        for i in range(repeats):
            run_num = repeats - i
            if eval_only is not False:
                if type(eval_only) is bool:
                    version_num = -run_num
                else:
                    version_num = eval_only + i
            else:
                version_num = None
            model_path = get_model_path(classifier_name, version_num, exp_name)
            model_folder = os.path.dirname(os.path.dirname(model_path))
            if "version_" in os.path.basename(model_folder):
                shutil.rmtree(model_folder)
            else:
                print("Not deleting model folder -- please check that folder name is correct")
                print("Folder path:", model_folder)
                print("Folder name:", os.path.basename(model_folder))

    return aggregated_results


def get_params(model_type, shallow_model_params, shallow_params, deep=True):
    mod = model_type.lower()
    if mod + "_params" not in shallow_model_params:
        if mod.lower() not in ["deepsvdd", "nn", "drocc", "abc"]:
        # if deep:
                raise KeyError(f'Model type "{model_type}" is not supported')
        else:
            print(model_type + " is not supported as shallow model.")
            if not deep:
                raise KeyError(f'Model type "{model_type}" is not supported')
        return
    for k, v in shallow_params[mod + "_params"].items():
        shallow_model_params[mod + "_params"][k] = v


def split_train_eval(
        x_training_real, y_training_real, num_real_training, x_test, y_test, val_split, class_label, dataset_name,
        synthetic_anom_ratio, synthetic_val_anom_constant, one_hot_col_len, binary_cols, delta,
        model_type, classifier_layers, rep_dim, activation, one_class, dropout, sigmoid_head, theory_nn,
        repeats, eval_only,
        epochs, weight_decay, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, patience, neg_labels,
        batch_size,
        loss_fn, fpr, shallow_params=dict(), deepsvdd_ae_ckpt_path=None, deepsvdd_ERM=False, deepsvdd_params=None,
        nn_params=None, quadratic_bump=False, drocc_params=None
):
    # Get shallow params
    shallow_model_params = {
        "svm_params": {"C": weight_decay},
        "rf_params":{"n_estimators": 100, "min_samples_split": 10, "min_samples_leaf": 1,
                "max_features": None, "max_leaf_nodes": None, "class_weight": "balanced_subsample"},
        "ocsvm_params": {},
        "isolationforest_params": {"n_estimators": 100},
        "lof_params": {"n_neighbors": 5, "leaf_size": 30, "novelty": True}
    }
    if type(model_type) is dict or type(model_type) is OrderedDict:
        for model_task, model_types in model_type.items():
            for model_type_for_indiv_task in model_types:
                get_params(model_type_for_indiv_task, shallow_model_params, shallow_params)
    elif model_type not in ["NN", "DeepSVDD", "CustomNN", "DROCC"]:
        get_params(model_type, shallow_model_params, shallow_params)
        # mod = model_type.lower()
        # if mod + "_params" not in shallow_model_params:
        #     raise KeyError(f'Model type "{model_type}" is not supported')
        # for k, v in shallow_params.items():
        #     shallow_model_params[mod + "_params"][k] = v

    results = []

    for i in range(repeats):
        run_num = repeats - i
        print("Run ", i + 1)
        # Resample train vs val split
        if num_real_training < 1:
            num_training = len(x_training_real)
            num_to_use = int(num_real_training * num_training)
            np.random.seed(seed=i + 135)
            indices = np.random.choice(num_training, num_to_use, replace=False)
            x_training_real_used = x_training_real[indices]
            y_training_real_used = y_training_real[indices]
        else:
            x_training_real_used = x_training_real
            y_training_real_used = y_training_real
        seed = 1234 + i
        num_anom = (y_training_real == 0).sum()
        if num_anom > 0:
            anom_split = 1 / num_anom
            print("number of anomalies during training:", num_anom)
            # for supervised AD: if anomalies are too few to be split into the val set,
            # adjust the val_split ratio to have 1 anom in val set
            if anom_split > val_split:
                val_split = anom_split
                print("new val_split:", val_split)
        x_train_real, y_train_real, x_val_real, y_val_real = validation_split(
            x_training_real_used, y_training_real_used, val_split=val_split, seed=seed)
        if num_anom > 0:
            while len(np.unique(y_train_real)) < 2 or len(np.unique(y_val_real)) < 2:
                # sup AD:
                # if there is only 1 class during training or validation, keep sampling until there are 2 classes
                seed -= 10
                print("new seed:", seed)
                x_train_real, y_train_real, x_val_real, y_val_real = validation_split(
                    x_training_real_used, y_training_real_used, val_split=val_split, seed=seed)
        print("val split is obtained")

        if type(model_type) is not dict and type(model_type) is not OrderedDict:
            classifier, classifier_name = get_classifier_and_name(
            model_type, classifier_layers=classifier_layers, rep_dim=rep_dim, activation=activation,
            one_class=one_class,
            dropout=dropout, sigmoid_head=sigmoid_head, quadratic_bump=quadratic_bump, seed=i)
        else:
            classifier = None
            classifier_name = None
        if theory_nn and model_type == "NN":
            classifier_name = "T" + classifier_name[1:]
        if model_type == "CustomNN":
            _model_type = "NN"
        else:
            _model_type = model_type
        print(classifier)

        # generate and combine synthetic anoms w seed 23+i
        # train eval model
        if eval_only is not False:
            if type(eval_only) is bool:
                version_num = -run_num
            else:
                version_num = eval_only + i
        else:
            version_num = None

        result = experiment_run(
            classifier, classifier_name, x_train_real, y_train_real, x_val_real, y_val_real, x_test, y_test,
            class_label,
            synthetic_anom_ratio, synthetic_val_anom_constant=synthetic_val_anom_constant,
            one_hot_col_len=one_hot_col_len, binary_cols=binary_cols, delta=delta, epochs=epochs, optimizer=optimizer,
            optimizer_params=optimizer_params, lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
            patience=patience, neg_labels=neg_labels,
            batch_size=batch_size, loss_fn=loss_fn,
            plot=True, fpr=fpr, normal_is_positive=False, seed=i, seed_anom_generation=23 + i, positive_class=0,
            model_type=_model_type, version_num=version_num, exp_name=dataset_name, **shallow_model_params,
            deepsvdd_ae_ckpt_path=deepsvdd_ae_ckpt_path, deepsvdd_ERM=deepsvdd_ERM, deepsvdd_params=deepsvdd_params,
            nn_params=nn_params, drocc_params=drocc_params)

        results.append(result)

    if type(model_type) is not dict and type(model_type) is not OrderedDict:
        results_dict = collate_results(
            results, result_cols=["precision", "recall", "f1", "average_precision", "auroc", "acc", "df_results"])
        aggregated_results = agg_results(results_dict)
        print_exp_agg_results(
            aggregated_results["precision"], aggregated_results["recall"], aggregated_results["f1"],
            aggregated_results["average_precision"], aggregated_results["auroc"], aggregated_results["acc"],
            aggregated_results["df_results"])

        return aggregated_results, classifier_name
    else:
        print("Aggregating Results of Composite Models...")
        # results is list of result
        # result is list of df_results, where each df_result corresponds to the composite result

        # Build result objects for each composite model
        composite_model_names = []
        for model_stage, model_list in model_type.items():
            if len(composite_model_names) == 0:
                composite_model_names = model_list
            else:
                new_names = []
                for name in composite_model_names:
                    for model_name in model_list:
                        new_names.append(f"{name}_{model_name}")
                composite_model_names = new_names

        results_of_composite_models = [list() for _ in range(len(composite_model_names))]
        for result in results:
            for i, composite_model_result in enumerate(result):
                results_of_composite_models[i].append(composite_model_result)
        composite_model_aggregated_list_of_results = []
        for model_name, composite_model_result_list in zip(composite_model_names, results_of_composite_models):
            agg_results_comp = get_mean_df_results(list_of_df_results=composite_model_result_list, key="df_results")
            composite_model_aggregated_list_of_results.append(agg_results_comp)
            print("\n***************************************************************")
            print(model_name)
            print_exp_agg_df_results(agg_results_comp)


        # print("Results Error")
        # print(e)
        return composite_model_aggregated_list_of_results, classifier_name


def set_hyperparameters(custom, hyperparam_dict):
    if hyperparam_dict['theory_nn']:
        hyperparam_dict['one_class'] = False
        hyperparam_dict['sigmoid_head'] = False
        hyperparam_dict['use_hinge'] = True
        if not custom:
            hyperparam_dict['activation'] = torch.nn.ReLU()
            # hyperparam_dict['activation'] = torch.nn.LeakyReLU()
            hyperparam_dict['weight_decay'] = 1e-2
            hyperparam_dict['patience'] = hyperparam_dict['epochs']

    if hyperparam_dict['weight_decay'] > 0:
        hyperparam_dict['optimizer'] = torch.optim.AdamW  # w weight decay
        hyperparam_dict["optimizer_params"]['weight_decay'] = hyperparam_dict['weight_decay']
    else:
        hyperparam_dict['optimizer'] = torch.optim.Adam
    # del hyperparam_dict['weight_decay']
    if hyperparam_dict['use_hinge']:
        # from torchmetrics import HingeLoss
        # hyperparam_dict['loss_fn'] = HingeLoss(task="binary")
        # from torchmetrics.classification import BinaryHingeLoss
        # hyperparam_dict['loss_fn'] = BinaryHingeLoss()
        hyperparam_dict['loss_fn'] = hinge_loss
        hyperparam_dict['neg_labels'] = False
        # hyperparam_dict['loss_fn'] = torch.nn.HingeEmbeddingLoss(margin=1.0)
        # hyperparam_dict['neg_labels'] = True
    else:
        hyperparam_dict['loss_fn'] = torch.nn.functional.binary_cross_entropy
        hyperparam_dict['neg_labels'] = False
    del hyperparam_dict['use_hinge']

    # for k, v in hyperparam_dict.items():
    #     # exec(f"{k} = v")
    #     globals()[k] = v

    return hyperparam_dict


def experiment_run(
        model, model_name, x_train_real, y_train_real, x_val_real, y_val_real, x_test, y_test, class_label,
        synthetic_anom_ratio, epochs, optimizer, optimizer_params, lr_scheduler, lr_scheduler_params, patience,
        synthetic_val_anom_constant=False, one_hot_col_len=[], binary_cols=True, delta=0., neg_labels=False,
        batch_size=128, loss_fn=torch.nn.functional.binary_cross_entropy, plot=True, fpr=0.05, normal_is_positive=True,
        version_num=None, seed=100, seed_anom_generation=23, model_type="NN", positive_class=0, exp_name='kdd',
        svm_params={"C": 1.}, rf_params={"n_estimators": 100, "min_samples_split": 10, "min_samples_leaf": 1,
        "max_features": None, "max_leaf_nodes": None, "class_weight": "balanced_subsample"},
        nn_params={"classifier":None, "classifier_name":"NN",
        "positive_class":0, "epochs":100, "optimizer":torch.optim.Adam, "optimizer_params":{"lr": 1e-3},
        "lr_scheduler":None, "lr_scheduler_params":dict(), "loss_fn":torch.nn.functional.binary_cross_entropy,
        "neg_labels":False, "patience":7, "seed":42, "exp_name":'kdd', "batch_size":1024},
        ocsvm_params={}, isolationforest_params={"n_estimators": 100}, lof_params={},
        deepsvdd_params={
        "rep_dim": [60], "epochs": 100, "batch_size": 32, "patience": 7,
        "optimizer": torch.optim.Adam, "optimizer_params": {'lr': 3e-4}, "lr_scheduler": None,
        "lr_scheduler_params": dict(), "exp_name": "kdd"},
        deepsvdd_ae_ckpt_path=None, deepsvdd_ERM=False,
        drocc_params={"lamda":1, "radius":0.2, "gamma":2.0, "only_ce_epochs":50,
               "ascent_step_size":0.001, "ascent_num_steps":50}):
    """
    Function to run the experiment for a given model
       Args:
           model: nn.Module or rep_dim (list of input and hidden dim)
           model_name: str or [dict, dict]
           x_train_real:
           y_train_real:
           x_val_real:
           y_val_real:
           x_test:
           y_test:
           class_label: dictionary of {"class_name": class_ID}
           synthetic_anom_ratio:
           epochs:
           optimizer:
           optimizer_params: dict of parameters to pass to the optimizer (e.g. {'lr': 1e-4})
           lr_scheduler:
           lr_scheduler_params:
           loss_fn: loss function
           one_hot_col_len: list of int, where i^th int corresponds to i^th categorical one-hot-encoded column
           binary_cols: bool of whether to sample bernoulli RVs for features that have only 0 or 1
           neg_labels: False for 0/1 label, True for -1/+1 label
           patience:
           batch_size:
           plot:
           fpr: allowable FPR (calculated from validation data) or None for threshold of 0.5
           normal_is_positive:
           version_num: None to train, int to use saved model
           seed:
           seed_anom_generation:
           model_type:
           positive_class:
           exp_name: name of experiment / dataset tested on

    Returns:

    """

    # label 0 for anom, 1 for normal/base class
    x_train, y_train, x_val, y_val = combine_real_synthetic(
        x_train_real, y_train_real, x_val_real, y_val_real, synthetic_anom_ratio, synthetic_val_anom_constant,
        one_hot_col_len, binary_cols, delta, seed_anom_generation=seed_anom_generation)

    # Methods
    quantile = fpr
    tpr = False

    if type(model_type) is dict or type(model_type) is OrderedDict:
        if deepsvdd_params is None:
            deepsvdd_params = dict()
        deepsvdd_params["ae_ckpt_path"] = deepsvdd_ae_ckpt_path
        deepsvdd_params["ERM"] = deepsvdd_ERM

        ########################################################
        ##### TODO: REMOVE THIS AFTER DEBUGGING PIPELINE  ######
        # x_train = x_train[:1024]
        # y_train = y_train[:1024]
        ########################################################

        all_results = train_eval_unsup_sup(
            x_train, y_train, x_val, y_val, x_test, y_test, class_label, pos_label=positive_class, batch_size=batch_size,
            normal_is_positive=normal_is_positive, eval_comments=False, quantile=quantile, tpr=tpr,
            plot=plot, model_type=model_type, seed=seed,
            svm=svm_params, rf=rf_params, nn=nn_params, ocsvm=ocsvm_params, iso_f=isolationforest_params, lof=lof_params,
            deepsvdd=deepsvdd_params)
        return all_results

    # Shallow Methods
    elif model_type not in ["NN", "DROCC", "ABC"]:
        # Deep SVDD
        if model_type.lower() == "deepsvdd":
            if version_num is None or deepsvdd_ae_ckpt_path is not None:
                # either train from scratch or train from pre-trained encoder
                if version_num is not None:
                    # train from pre-trained encoder
                    if version_num < 0:
                        version_num = len(os.listdir(deepsvdd_ae_ckpt_path)) + version_num
                    ae_ckpt_path = os.path.join(deepsvdd_ae_ckpt_path, f"version_{version_num}/checkpoints")
                    ckpt = os.path.join(ae_ckpt_path, os.listdir(ae_ckpt_path)[-1])
                else:
                    ckpt = None
                precision, recall, f1, average_precision, auroc, acc, df_results, threshold = train_eval_deep_svdd(
                    x_train, y_train, x_val, y_val, x_test, y_test, model,
                    class_label, pos_label=positive_class, epochs=epochs, batch_size=batch_size, patience=patience,
                    optimizer=optimizer, optimizer_params=optimizer_params,
                    lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
                    normal_is_positive=normal_is_positive, eval_comments=False, quantile=quantile, tpr=tpr,
                    plot=True, exp_name=exp_name, seed=seed, ae_ckpt_path=ckpt, ERM=deepsvdd_ERM)

                # (best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results,
                #  threshold) = train_eval(
                #     model, model_name, train_loader, val_loader, test_loader, testing_datasets=testing_datasets,
                #     class_label=class_label, positive_class=positive_class, epochs=epochs, optimizer=optimizer,
                #     optimizer_params=optimizer_params, lr_scheduler=lr_scheduler,
                #     lr_scheduler_params=lr_scheduler_params,
                #     loss_fn=loss_fn,
                #     patience=patience, neg_labels=neg_labels, quantile=quantile, tpr=tpr,
                #     normal_is_positive=normal_is_positive, plot=plot, plot_name=model_name,
                #     seed=seed, eval_comments=False, exp_name=exp_name)

            else:
                (val_loader,), testing_datasets, test_loader = np_to_dataloader(
                    [x_val], [y_val], x_test, y_test, batch_size)
                model_name = f"{model_name}{len(model)}_{model[-1]}"
                experiment_path = os.path.join("logs/", exp_name, model_name)
                if version_num < 0:
                    version_num = len(os.listdir(experiment_path)) + version_num
                model_folder = os.path.join(experiment_path,  f"version_{version_num}")
                ckpt_path = os.path.join(model_folder, 'checkpoints')
                best_model_ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[-1])
                print("Loading SVDD Model from", best_model_ckpt_path)
                # encoder = PL_Model.load_from_checkpoint(
                #     best_model_ckpt_path, backbone=encoder, classifier=decoder, use_hidden_layer_of_backbone=False
                # ).feature_extractor.to(device)
                # centre =
                rep_dim = model
                encoder = build_model(rep_dim, bias=False)
                centre_path = os.path.join(model_folder, "centre.pt")
                centre = torch.load(centre_path)
                model = DeepSVDD.load_from_checkpoint(best_model_ckpt_path, encoder=encoder, centre=centre)

                precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
                    model, val_loader, test_loader, testing_datasets, class_label, eval_comments=False,
                    quantile=quantile, tpr=tpr, pos_label=positive_class, normal_is_positive=normal_is_positive,
                    plot=plot, plot_threshold=False, plot_name=model_name)
                # (best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results,
                #  threshold) = eval_run(model, model_name, val_loader, test_loader, testing_datasets, class_label,
                #                        loss_fn=loss_fn, quantile=quantile, tpr=tpr,
                #                        pos_label=positive_class, normal_is_positive=False,
                #                        plot=True, eval_comments=False, version_num=version_num, exp_name=exp_name)
        else:
            precision, recall, f1, average_precision, auroc, acc, df_results, threshold = train_eval_shallow(
                x_train, y_train, x_val, y_val, x_test, y_test, class_label, pos_label=positive_class,
                normal_is_positive=normal_is_positive, eval_comments=False, quantile=quantile, tpr=tpr,
                plot=True, model_type=model_type, seed=seed,
                svm=svm_params, ocsvm=ocsvm_params, iso_f=isolationforest_params, lof=lof_params)
        # x_train, y_train, x_test, y_test, fpr=fpr, kernel='rbf')
        # print("Degree 3 Polynomial Kernel SVM")
        # pr_auc_svm_poly, roc_auc_svm_poly = train_eval_svm(x_train, y_train, x_test, y_test, fpr=fpr, kernel='poly',
        #                                                    degree=3)

        # return pr_auc_svm, roc_auc_svm

    else:
        # NN or DROCC
        np.random.seed(0)
        np.random.shuffle(x_train)
        np.random.seed(0)
        np.random.shuffle(y_train)
        # train_loader = get_dataloader(x_train, y_train, batch_size=batch_size)
        # val_loader = get_dataloader(x_val, y_val, batch_size=batch_size)
        #
        # # Separate into different labels
        # # convert base class (class 0) to 1 and others/anomalies (class 1/2/3/...) to 0
        # testing_datasets = get_datasets_by_label(x_test, y_test)
        # test_loader = get_dataloader(x_test, (y_test == 0), batch_size=batch_size)
        (train_loader, val_loader), testing_datasets, test_loader = np_to_dataloader(
            [x_train, x_val], [y_train, y_val], x_test, y_test, batch_size)

        # classifier_layers = kwargs.get("classifier_layers", 3)
        # rep_dim = kwargs.get("rep_dim", 10)
        # epochs = kwargs.get("epochs", 100)
        # optimizer = kwargs.get("optimizer", torch.optim.Adam)
        # lr = kwargs.get("lr", 1e-3)
        # patience = kwargs.get("patience", 7)
        # plot = kwargs.get("plot", True)
        # exp_num = kwargs.get("exp_num", 1)
        # one_class = False
        if model_type == "DROCC":
            drocc_params = drocc_params
        else:
            drocc_params = None

        if version_num is None:

            (best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results,
             threshold) = train_eval(
                model, model_name, train_loader, val_loader, test_loader, testing_datasets=testing_datasets,
                class_label=class_label, positive_class=positive_class, epochs=epochs, optimizer=optimizer,
                optimizer_params=optimizer_params, lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
                loss_fn=loss_fn,
                patience=patience, neg_labels=neg_labels, quantile=quantile, tpr=tpr,
                normal_is_positive=normal_is_positive, plot=plot, plot_name=model_name,
                seed=seed, eval_comments=False, exp_name=exp_name, drocc_params=drocc_params)

        else:
            (best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results,
             threshold) = eval_run(model, model_name, val_loader, test_loader, testing_datasets, class_label,
                                   loss_fn=loss_fn, quantile=quantile, tpr=tpr,
                                   pos_label=positive_class, normal_is_positive=False,
                                   plot=True, eval_comments=False, version_num=version_num, exp_name=exp_name,
                                   drocc_params=drocc_params)

        print("Best NN Model:", best_model_ckpt_path)

        # # NN (One-Class)
        # one_class = True
        #
        # (best_model_oc, best_model_ckpt_path_oc, precision_oc, recall_oc, f1_oc, average_precision_oc, auroc_oc,
        #  df_results_oc, threshold_oc) = train_eval(
        #     classifier_layers, rep_dim, train_loader, val_loader, test_loader, testing_datasets=None, class_label=None,
        #     epochs=epochs, optimizer=optimizer, lr=lr, patience=patience, quantile=quantile, tpr=tpr, plot=plot,
        #     seed=seed, one_class=one_class, eval_comments=False, exp_num=exp_num)
        #
        # print("Best NN (OC) Model:", best_model_ckpt_path_oc)

    return precision, recall, f1, average_precision, auroc, acc, df_results


def train_model(model_type, x_train, y_train, x_val, y_val, x_test, y_test, quantile=0.05, seed=100,
                svm=None, rf=None, nn=None, ocsvm=None, iso_f=None, lof=None, deepsvdd=None, drocc=None,
                unsup_use_all_as_normal=False):
    if model_type == "SVM":
        print("RBF Kernel SVM")
        model = SVC(**svm)
        model.fit(x_train, y_train)
    elif model_type == "RF":
        print("Random Forest")
        model = RandomForestClassifier(random_state=seed, **rf)
        model.fit(x_train, y_train)
    elif model_type.lower() in ["nn", "drocc", "abc"]:
        np.random.seed(0)
        np.random.shuffle(x_train)
        np.random.seed(0)
        np.random.shuffle(y_train)
        # train_loader = get_dataloader(x_train, y_train, batch_size=batch_size)
        # val_loader = get_dataloader(x_val, y_val, batch_size=batch_size)
        #
        # # Separate into different labels
        # # convert base class (class 0) to 1 and others/anomalies (class 1/2/3/...) to 0
        # testing_datasets = get_datasets_by_label(x_test, y_test)
        # test_loader = get_dataloader(x_test, (y_test == 0), batch_size=batch_size)
        (train_loader, val_loader), testing_datasets, test_loader = np_to_dataloader(
            [x_train, x_val], [y_train, y_val], x_test, y_test, nn.get("batch_size", 1024))
        if model_type.lower() == "drocc":
            kwargs = drocc
        elif model_type.lower() == "abc":
            kwargs = {"abc": True}
        else:
            kwargs = dict()
        model, model_ckpt_path = train_nn(train_loader=train_loader, val_loader=val_loader, **nn, **kwargs)

    elif model_type == "OCSVM":
        print("RBF Kernel OCSVM")
        if "nu" in ocsvm:
            model = OneClassSVM(**ocsvm)
        else:
            if quantile is None:
                nu = 1e-10
            else:
                nu = quantile
            model = OneClassSVM(nu=nu, **ocsvm)
        if unsup_use_all_as_normal:
            model.fit(x_train)
        else:
            model.fit(x_train[y_train == 1])
    elif model_type == "IsolationForest" or model_type == "IsoF":
        print("Isolation Forest")
        if "contamination" in iso_f:
            model = IsolationForest(random_state=seed, **iso_f)
        else:
            if quantile is None:
                contamination = 'auto'
            else:
                contamination = quantile
            model = IsolationForest(random_state=seed, contamination=contamination, **iso_f)

        if unsup_use_all_as_normal:
            model.fit(x_train)
        else:
            model.fit(x_train[y_train == 1])
    elif model_type == "LOF":
        print("LOF")
        if "contamination" in lof:
            model = LOF(**lof)
        else:
            if quantile is None:
                contamination = 'auto'
            else:
                contamination = quantile
            model = LOF(contamination=contamination, **lof)

        model.fit(x_train[y_train == 1])
    elif model_type.lower() == "deepsvdd":
        model, testing_loader, testing_datasets = train_deep_svdd(
        x_train, y_train, x_val, y_val, x_test, y_test, seed=seed,
            unsup_use_all_as_normal=unsup_use_all_as_normal, **deepsvdd)
    else:
        raise ValueError("Model type not supported:", model_type)
    return model


def get_pred_args(model_type, using_test_data, metrics_by_class, change_y_anom_id_to_0_1, pred_data):
    """

    Args:
        model_type: str of model type, or device (cpu or cuda)
        using_test_data: bool of whether test data is being passed through
        metrics_by_class: bool of whether metrics are being calculated for each class (vs just overall)
        change_y_anom_id_to_0_1: bool of whether y anomaly id (0, 1/2/3/...) should be converted to (1, 0)
        pred_data: tup of data to be predicted (non torch data e.g. np array, torch dataloader)

    Returns:

    """
    keyword_args = dict()

    if model_type.lower() in ["nn", "deepsvdd", "drocc", "cuda"]:
        keyword_args["torch_model"] = True
        keyword_args["testing_datasets"] = pred_data[1]
    else:
        keyword_args["torch_model"] = False
        keyword_args["testing_datasets"] = pred_data[0]

    if using_test_data:
        # keyword_args["eval_comments"] = eval_comments
        if metrics_by_class and change_y_anom_id_to_0_1:
            keyword_args["label_is_map"] = True
        else:
            keyword_args["label_is_map"] = False

    else:
        keyword_args["label_is_map"] = False
        # keyword_args["eval_comments"] = False
    return keyword_args


def train_pred(model_type, x_train, y_train, x_val, y_val, x_test, y_test, pred_data,
        class_label, pos_label=1, quantile=0.05, seed=100,
        svm=None, rf=None, nn=None, ocsvm=None, iso_f=None, lof=None, deepsvdd=None, drocc=None,
               unsup_use_all_as_normal=False, make_positive=False):

    model_trained = train_model(
        model_type, x_train, y_train, x_val, y_val, x_test, y_test,
        quantile=quantile, seed=seed,
        svm=svm, rf=rf, nn=nn, ocsvm=ocsvm, iso_f=iso_f, lof=lof, deepsvdd=deepsvdd, drocc=drocc,
        unsup_use_all_as_normal=unsup_use_all_as_normal)
    # TODO: Add validation data

    # Pred test data
    pred_kwargs = get_pred_args(
        model_type, using_test_data=True, metrics_by_class=True, change_y_anom_id_to_0_1=False,
        pred_data=pred_data)
    # 2 np arrays. y_true_test is [0,0,...,0, 1,1,...,1, 2,2,..., 2, 3,3,...3]
    y_pred_test, y_true_test = pred_per_class(
        model_trained, class_label=class_label, pos_label=pos_label, **pred_kwargs)
    # y_pred_test, y_true_test = predict_from_loader(model_trained, pos_label=pos_label)
    # models[model_type] =
    if make_positive:
        min_element = np.min(y_pred_test)
        # here, min_element is a scalar, but can be expanded to include arrays
        # if min_element <= 0:
        y_pred_test -= min_element
        # y_pred_test /= np.var(y_pred_test)
        # normalise to btw 0 and 1
        y_pred_test /= np.max(y_pred_test)
        # add by small amount so wont get n x 0 = 0
        y_pred_test += 1e-6

    return y_pred_test


def train_eval_unsup_sup(x_train, y_train, x_val, y_val, x_test, y_test, class_label, pos_label=1, batch_size=1024,
                         normal_is_positive=True, eval_comments=False, quantile=0.05, tpr=False,
                         plot=True, model_type={"unsup_AD": [], "sup_BC": []}, seed=100,
                         svm=None, rf=None, nn=None, ocsvm=None, iso_f=None, lof=None, deepsvdd=None, drocc=None):


    # normal_label = get_normal_label(pos_label=pos_label, normal_is_positive=normal_is_positive)
    base_class_label = 0
    y_true_normal = (y_test == base_class_label)
    # classes = np.unique(y_test)
    # classes.sort()
    # convert base class (class 0) to 1 and others/anomalies (class 1/2/3/...) to 0
    testing_datasets = get_datasets_by_label(x_test, y_test, torch_dataset=False)

    _, testing_datasets_torch, test_loader_torch = np_to_dataloader(
        None, None, x_test, y_test, batch_size)
    pred_data = (testing_datasets, testing_datasets_torch)



    # if type(model_type) is dict or type(model_type) is OrderedDict:
    unsup_model_types = model_type["unsup_AD"]
    sup_model_types = model_type["sup_BC"]
    unsup_preds = []
    sup_preds = []
    # Train and Pred Models
    for unsup_model_type in unsup_model_types:
        # unsup_model_trained = train_model(
        #     unsup_model_type, x_train, y_train, x_val, y_val, x_test, y_test,
        #     quantile=quantile, seed=seed,
        #     svm=svm, rf=rf, nn=nn, ocsvm=ocsvm, iso_f=iso_f, lof=lof, deepsvdd=deepsvdd)
        # # TODO: Add validation data
        #
        # # Pred test data
        # pred_kwargs = get_pred_args(
        #     unsup_model_type, using_test_data=True, metrics_by_class=True, change_y_anom_id_to_0_1=False,
        #     pred_data=pred_data)
        # # 2 np arrays. y_true_test is [0,0,...,0, 1,1,...,1, 2,2,..., 2, 3,3,...3]
        # y_pred_test, y_true_test = pred_per_class(
        #     unsup_model_trained, class_label=class_label, pos_label=pos_label, **pred_kwargs)
        # # y_pred_test, y_true_test = predict_from_loader(unsup_model_trained, pos_label=pos_label)

        # scale all results so that they are all non-negative (otherwise, negative times negative is positive)
        #       use make_positive option
        pred = train_pred(
            unsup_model_type, x_train, y_train, x_val, y_val, x_test, y_test, pred_data,
            class_label=class_label, pos_label=pos_label, quantile=quantile, seed=seed,
            svm=svm, rf=rf, nn=nn, ocsvm=ocsvm, iso_f=iso_f, lof=lof, deepsvdd=deepsvdd, drocc=drocc,
            unsup_use_all_as_normal=True, make_positive=True)
        unsup_preds.append(pred)
    for sup_model_type in sup_model_types:
        pred = train_pred(
            sup_model_type, x_train, y_train, x_val, y_val, x_test, y_test, pred_data,
            class_label=class_label, pos_label=pos_label, quantile=quantile, seed=seed,
            svm=svm, rf=rf, nn=nn, ocsvm=ocsvm, iso_f=iso_f, lof=lof, deepsvdd=deepsvdd, drocc=drocc,
            make_positive=True)
        sup_preds.append(pred)
        # sup_models.append(
        #     train_model(sup_model_type, x_train, y_train, x_val, y_val, x_test, y_test,
        #                 quantile=quantile, seed=seed,
        #                 svm=svm, rf=rf, nn=nn, ocsvm=ocsvm, iso_f=iso_f, lof=lof, deepsvdd=deepsvdd)
        # )


    ####### Eval multiple unsup-sup model pairs
    # Multiply scores across all models

    unsup_preds_reshaped = np.array(unsup_preds)[:, np.newaxis, :]
    sup_preds_reshaped = np.array(sup_preds)[np.newaxis, :, :]
    # [unsup0 * sup0, unsup0 * sup1, ..., unsup0 * supn, unsup1 * sup0,...,unsup1 * supn, ..., unsupm * sup0,...,unsupm * supn]
    composite_results = (unsup_preds_reshaped * sup_preds_reshaped).reshape(len(unsup_preds) * len(sup_preds), -1)

    # Eval the composite score
    i = 0
    all_results = []
    for unsup_model_type in unsup_model_types:
        for sup_model_type in sup_model_types:
            df_results = eval_preds_per_class(composite_results[i], y_test, class_label, threshold=0.0, normal_is_positive=normal_is_positive,
                         eval_comments=eval_comments, plot=plot, plot_name=f"{unsup_model_type} with {sup_model_type}")
            all_results.append(df_results)
            i += 1

    # precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
    #     model, (x_val, y_val), (x_test, y_true_normal), testing_datasets, class_label,
    #     eval_comments=eval_comments,
    #     quantile=quantile, tpr=tpr, pos_label=pos_label, normal_is_positive=normal_is_positive, device=None, plot=plot,
    #     plot_threshold=False, plot_name=model_type,
    #     **kwargs)


    # if plot:
    #     if normal_is_positive:
    #         classes_plot = ["Normal", "Anomaly"]
    #     else:
    #         classes_plot = ["Anomaly", "Normal"]
    #     _, _ = plot_metrics(y_true_anom, y_anom_score, classes=classes_plot, fpr=fpr, name="SVM")

    return all_results



def train_eval_shallow(x_train, y_train, x_val, y_val, x_test, y_test, class_label, pos_label=1,
                       normal_is_positive=True, eval_comments=False, quantile=0.05, tpr=False,
                       plot=True, model_type="SVM", seed=100, svm=None, ocsvm=None, iso_f=None, lof=None, **kwargs):
    if model_type == "SVM":
        print("RBF Kernel SVM")
        model = SVC(**svm)
        model.fit(x_train, y_train)
    elif model_type == "OCSVM":
        print("RBF Kernel OCSVM")
        if "nu" in ocsvm:
            model = OneClassSVM(**ocsvm)
        else:
            if quantile is None:
                nu = 0.0000000001
            else:
                nu = quantile
            model = OneClassSVM(nu=nu, **ocsvm)
        model.fit(x_train[y_train == 1])
    elif model_type == "IsolationForest" or model_type == "IsoF":
        print("Isolation Forest")
        if "contamination" in iso_f:
            model = IsolationForest(random_state=seed, **iso_f)
        else:
            if quantile is None:
                contamination = 'auto'
            else:
                contamination = quantile
            model = IsolationForest(random_state=seed, contamination=contamination, **iso_f)

        model.fit(x_train[y_train == 1])
    elif model_type == "LOF":
        print("LOF")
        if "contamination" in lof:
            model = LOF(**lof)
        else:
            if quantile is None:
                contamination = 'auto'
            else:
                contamination = quantile
            model = LOF(contamination=contamination, **lof)

        model.fit(x_train[y_train == 1])
    else:
        raise ValueError("Model type not supported:", model_type)
    # normal_label = get_normal_label(pos_label=pos_label, normal_is_positive=normal_is_positive)
    base_class_label = 0
    y_true_normal = (y_test == base_class_label)
    # classes = np.unique(y_test)
    # classes.sort()
    # convert base class (class 0) to 1 and others/anomalies (class 1/2/3/...) to 0
    testing_datasets = get_datasets_by_label(x_test, y_test, torch_dataset=False)

    precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
        model, (x_val, y_val), (x_test, y_true_normal), testing_datasets, class_label,
        eval_comments=eval_comments,
        quantile=quantile, tpr=tpr, pos_label=pos_label, normal_is_positive=normal_is_positive, device=None, plot=plot,
        plot_threshold=False, plot_name=model_type,
        **kwargs)

    # if plot:
    #     if normal_is_positive:
    #         classes_plot = ["Normal", "Anomaly"]
    #     else:
    #         classes_plot = ["Anomaly", "Normal"]
    #     _, _ = plot_metrics(y_true_anom, y_anom_score, classes=classes_plot, fpr=fpr, name="SVM")

    return precision, recall, f1, average_precision, auroc, acc, df_results, threshold


def train_deep_svdd(
        x_train, y_train, x_val, y_val, x_test, y_test, rep_dim,
        unsup_use_all_as_normal=False,
        epochs=100, batch_size=128, patience=7,
        optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3}, lr_scheduler=None, lr_scheduler_params=dict(),
        exp_name='kdd', seed=100, ae_ckpt_path=None, ERM=False):
    model_name = "DeepSVDD"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    np.random.seed(0)
    np.random.shuffle(x_train)
    if unsup_use_all_as_normal:
        x_train_normal = x_train
        x_val_normal = x_val
    else:
        x_train_normal = x_train[y_train == 1]
        x_val_normal = x_val[y_val == 1]
    train_loader = get_dataloader(x_train_normal, x_train_normal, batch_size=batch_size)
    val_loader = get_dataloader(x_val_normal, x_val_normal, batch_size=batch_size)

    # AE model
    encoder = build_model(rep_dim, bias=False)
    decoder = build_model(list(reversed(rep_dim)), bias=False)
    model = PL_Model(encoder, decoder, optimizer=optimizer, optimizer_params=optimizer_params,
                     lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
                     loss_fn=torch.nn.functional.mse_loss, freeze_backbone=False, use_hidden_layer_of_backbone=False,
                     seed=seed)
    experiment_path = os.path.join("logs", exp_name, f"{model_name}_AE{len(rep_dim)}_{rep_dim[-1]}")
    if ae_ckpt_path:
        best_encoder_ckpt_path = ae_ckpt_path
        # model_folder = os.path.dirname(ae_ckpt_path)
    else:
        # train AE
        early_stopping = EarlyStopping('val_loss', patience=patience)
        os.makedirs(experiment_path, exist_ok=True)
        version_num = len(os.listdir(experiment_path))
        model_folder = os.path.join(experiment_path, f"version_{version_num}")
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(model_folder, 'checkpoints'))
        trainer = L.Trainer(max_epochs=epochs, deterministic="warn", enable_progress_bar=True,
                            logger=TensorBoardLogger(save_dir=model_folder, name=f"tensorboard_logging"), log_every_n_steps=10,
                            callbacks=[early_stopping, checkpoint_callback])
        trainer.fit(model, train_loader, val_loader)
        best_encoder_ckpt_path = checkpoint_callback.best_model_path
    print("best_encoder_ckpt_path", best_encoder_ckpt_path)
    encoder = PL_Model.load_from_checkpoint(
        best_encoder_ckpt_path, backbone=encoder, classifier=decoder, use_hidden_layer_of_backbone=False
    ).feature_extractor.to(device)

    # Train DeepSAD
    if ERM:
        (train_loader, val_loader), testing_datasets, test_loader = np_to_dataloader(
            [x_train, x_val],
            [y_train, y_val], x_test, y_test, batch_size)
    # Train DeepSVDD
    else:
        (train_loader, val_loader), testing_datasets, test_loader = np_to_dataloader(
            [x_train_normal, x_val_normal],
            [np.ones_like(x_train_normal), np.ones_like(x_val_normal)], x_test, y_test, batch_size)
    early_stopping = EarlyStopping('val_loss', patience=patience)
    experiment_path = os.path.join("logs", exp_name, f"{model_name}{len(rep_dim)}_{rep_dim[-1]}")
    os.makedirs(experiment_path, exist_ok=True)
    version_num = len(os.listdir(experiment_path))
    model_folder = os.path.join(experiment_path, f"version_{version_num}")
    checkpoint_callback = ModelCheckpoint(dirpath=f'{experiment_path}/version_{version_num}/checkpoints')
    encoder.train()

    centre_path = os.path.join(model_folder, "centre.pt")
    if not os.path.exists(centre_path):
        encoder.eval()
        pred_recon = []
        with torch.no_grad():
            for x, _ in train_loader:
                pred_recon.append(encoder(x.to(device)))
            for x, _ in val_loader:
                pred_recon.append(encoder(x.to(device)))

        centre = torch.mean(torch.cat(pred_recon), dim=0).to(device)
        os.makedirs(model_folder, exist_ok=True)
        torch.save(centre, centre_path)
    else:
        centre = torch.load(centre_path)
    print("centre:", centre.size())

    deep_svdd = DeepSVDD(encoder, centre, optimizer=optimizer, optimizer_params=optimizer_params,
                         lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params, seed=seed).to(device)
    trainer = L.Trainer(max_epochs=epochs, deterministic="warn", enable_progress_bar=True,
                        logger=TensorBoardLogger(save_dir=model_folder, name=f"tensorboard_logging"), log_every_n_steps=10,
                        callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(deep_svdd, train_loader, val_loader)

    best_model_ckpt_path = checkpoint_callback.best_model_path
    best_model = DeepSVDD.load_from_checkpoint(best_model_ckpt_path, encoder=encoder, centre=centre).to(device)

    return best_model, test_loader, testing_datasets


def train_eval_deep_svdd(
        x_train, y_train, x_val, y_val, x_test, y_test, rep_dim,
        class_label, pos_label=1, epochs=100, batch_size=128, patience=7,
        optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3}, lr_scheduler=None, lr_scheduler_params=dict(),
        normal_is_positive=True, eval_comments=False, quantile=0.05, tpr=False,
        plot=True, exp_name='kdd', seed=100, ae_ckpt_path=None, ERM=False, **kwargs):
    model_name = "DeepSVDD"
    device = "cuda" if torch.cuda.is_available() else "cpu"


    best_model, test_loader, testing_datasets = train_deep_svdd(
        x_train, y_train, x_val, y_val, x_test, y_test, rep_dim,
        epochs=epochs, batch_size=batch_size, patience=patience,
        optimizer=optimizer, optimizer_params=optimizer_params, lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
        exp_name=exp_name, seed=seed, ae_ckpt_path=ae_ckpt_path, ERM=ERM)
    # Evaluate on Dataset
    print("Evaluating model...")
    # can see validation data with synthetic random data
    val_loader = get_dataloader(x_val, y_val, batch_size=batch_size)
    precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
        best_model, val_loader, test_loader, testing_datasets, class_label, eval_comments=eval_comments,
        quantile=quantile, tpr=tpr, pos_label=pos_label, normal_is_positive=normal_is_positive, plot=plot,
        plot_threshold=False, plot_name=model_name, device=device,
        **kwargs)

    return precision, recall, f1, average_precision, auroc, acc, df_results, threshold


def get_df_results(results):
    df_results = pd.DataFrame(data=results).T
    mean, std = df_results.mean(axis=1), df_results.std(axis=1)
    df_results["mean"] = mean
    df_results["std"] = std

    return df_results


def get_classifier_and_name(model_type, classifier_layers=None, rep_dim=None, activation=torch.nn.LeakyReLU(),
                            dropout=0, one_class=True, sigmoid_head=True, quadratic_bump=False, seed=0):
    if model_type in ["NN", "DROCC", "ABC"]:
        # create model w seed i
        L.seed_everything(seed, workers=True)
        if one_class:
            classifier_name = f"OC{classifier_layers}"
        else:
            classifier_name = f"BC{classifier_layers}"
        if model_type == "DROCC":
            classifier_name = "DR" + classifier_name[2:]
        if model_type == "ABC":
            classifier_name = "AC" + classifier_name[2:]
            classifier = build_model(rep_dim + list(reversed(rep_dim[:-1])), bias_last=False)
        else:
            classifier = build_classifier(
                classifier_layers=classifier_layers, rep_dim=rep_dim, activation=activation, dropout=dropout,
                one_class=one_class, sigmoid_head=sigmoid_head, quadratic_bump=quadratic_bump, seed=None)
    elif model_type in ["SVM", "OCSVM", "IsolationForest", "IsoF", "LOF", "DeepSVDD"]:
        # use shallow method
        classifier = rep_dim
        classifier_name = model_type
    elif model_type == "CustomNN":
        classifier = rep_dim(seed=seed)
        classifier_name = model_type
    else:
        raise Exception("Model type not supported")

    return classifier, classifier_name
