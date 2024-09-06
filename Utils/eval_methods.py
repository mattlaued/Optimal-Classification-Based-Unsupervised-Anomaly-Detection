import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_curve, auc, RocCurveDisplay, \
    precision_recall_curve, accuracy_score
import torch
import matplotlib.pyplot as plt
import pandas as pd
import lightning as L
import os

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from Utils.model_methods import PL_Model


def calculate_random(testing_datasets, pos_class=0, class_label={}):
    print("Baseline Average Precision (AP) / Area under Precision-Recall Curve (AUPR)")
    num_classes = len(testing_datasets)
    classes = list(range(num_classes))
    if class_label is None:
        class_label = list(range(num_classes))
    else:
        class_label = {num: name for name, num in class_label.items()}
    count = {i: len(testing_datasets[i]) for i in classes}
    classes.remove(pos_class)
    total_neg = sum([count[i] for i in classes])
    total_pos = count[pos_class]

    overall_aupr = total_pos / (total_neg + total_pos)
    print(f"Baseline Overall AP: ", overall_aupr)

    auprs = {"Overall": overall_aupr}

    for neg in classes:
        num_neg = count[neg]
        aupr = total_pos / (total_pos + num_neg)
        print(f"Baseline AP {neg} {class_label[neg]:>10}:", aupr)
        auprs[neg] = aupr

    return auprs


def train_eval(classifier, classifier_name, train_loader, val_loader, test_loader, testing_datasets, class_label,
               positive_class=0, epochs=100,
               optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3}, lr_scheduler=None, lr_scheduler_params=dict(),
               loss_fn=torch.nn.functional.binary_cross_entropy, neg_labels=False, patience=7, quantile=0.05, tpr=False,
               normal_is_positive=True, plot=True, seed=42, eval_comments=False, plot_name=None, exp_name='kdd', **kwargs):
    """
    Train and evaluate model.
    Args:
        # backbone_model: name of pretrained encoder model to use, or torch.nn.Module model
        # classifier_layers: number of layers in classifier head
        train_loader:
        val_loader:
        test_loader:
        # testing_datasets:
        # class_label:
        epochs:
        optimizer:
        optimizer_params: dict of parameters to pass to the optimizer (e.g. {'lr': 1e-4})
        # fast_dev_run:
        loss_fn: loss function
        neg_labels: False for 0/1 label, True for -1/+1 label
        quantile: quantile for threshold generation
        tpr: Whether to monitor TPR or FPR for threshold generation
        normal_is_positive: True if normal class is positive class. else, False
        plot: plot figures
        seed: random seed
        # one_class: True to train one-class classifier, False to train binary (two-class) classifier
        # nrf_train: whether training and evaluating with NRF or not (flips labels)
        eval_comments:
        kwargs: keyword arguments
    :return:

    """

    # Build Model
    # backbone, backbone_name, rep_dim = build_backbone(backbone_model, **kwargs)

    # L.seed_everything(seed, workers=True)
    # if one_class:
    #     classifier_name = f"C{classifier_layers}"
    # else:
    #     classifier_name = f"BC{classifier_layers}"
    # classifier = build_classifier(classifier_layers=classifier_layers, rep_dim=rep_dim, activation=torch.nn.LeakyReLU(),
    #                               one_class=one_class,
    #                               seed=None)

    # Train Model
    print("Training model...")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_name = classifier_name
    # print(model_name)
    early_stopping = EarlyStopping('val_loss', patience=patience)
    # need to include version num
    # exp_num = kwargs.get("exp_num", 1)
    experiment_path = "logs/" + exp_name + "/" + model_name
    os.makedirs(experiment_path, exist_ok=True)
    version_num = len(os.listdir(experiment_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{experiment_path}/version_{version_num}/checkpoints')
    # , monitor="val_loss")

    model_pl = PL_Model(
        backbone=None, classifier=classifier, positive_class=positive_class, optimizer=optimizer,
        optimizer_params=optimizer_params, lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params,
        loss_fn=loss_fn, neg_labels=neg_labels, seed=seed, device=device)

    trainer = L.Trainer(max_epochs=epochs, deterministic="warn", enable_progress_bar=True,
                        logger=TensorBoardLogger(save_dir="logs/", name=model_name), log_every_n_steps=10,
                        callbacks=[early_stopping, checkpoint_callback])
    trainer.fit(model_pl, train_loader, val_loader)

    # # Get best model
    # experiment_path = "logs/" + model_name
    # directory = os.listdir(experiment_path)
    # if len(directory) == 0:
    #     version_num = "version_0"
    # else:
    #     version_nums = [int(v.split("_")[-1]) for v in directory]
    # can also use
    # max([os.path.join(experiment_path, basename) for basename in os.listdir(experiment_path)], key=os.path.getctime)
    # version_num = "version_" + str(len(os.listdir(experiment_path)) - 1)
    # ckpt_path = os.path.join(experiment_path, version_num, "checkpoints")
    # best_model_ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[-1])
    best_model_ckpt_path = checkpoint_callback.best_model_path
    best_model = PL_Model.load_from_checkpoint(best_model_ckpt_path, backbone=None, classifier=classifier)
    # Evaluate on Dataset
    print("Evaluating model...")
    precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
        best_model, val_loader, test_loader, testing_datasets, class_label, eval_comments=eval_comments,
        quantile=quantile, tpr=tpr, pos_label=positive_class, normal_is_positive=normal_is_positive, plot=plot,
        plot_threshold=False, plot_name=plot_name,
        **kwargs)

    # print("Threshold:", threshold)
    # print("Overall precision, recall, f1, average_precision", precision, recall, f1, average_precision)
    # print(df_results)

    return best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results, threshold


def eval_run(classifier, model_name, val_loader, test_loader, testing_datasets, class_label,
             quantile=0.05, tpr=False, pos_label=0, normal_is_positive=False, plot=True,
             eval_comments=False, version_num=None, exp_name='kdd', **kwargs):

    # robustness = "_NRF" if nrf_train else ""

    # Get best model
    experiment_path = "logs/" + exp_name + "/" + model_name + "/"
    # can also use
    # max([os.path.join(experiment_path, basename) for basename in os.listdir(experiment_path)], key=os.path.getctime)
    if version_num is None:
        version_num = len(os.listdir(experiment_path)) - 1
    elif version_num < 0:
        version_num = len(os.listdir(experiment_path)) + version_num

    ckpt_path = os.path.join(experiment_path, "version_" + str(version_num), "checkpoints")
    best_model_ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[-1])
    print("Loading model from:", best_model_ckpt_path)

    # best_model_ckpt_path = checkpoint_callback.best_model_path
    best_model = PL_Model.load_from_checkpoint(
        best_model_ckpt_path, backbone=None, classifier=classifier)

    # Evaluate on Dataset
    print("Evaluating model...")
    # precision, recall, f1, average_precision, df_results, threshold = eval_model(
    #     best_model, val_loader, test_loader, testing_datasets, class_label, eval_comments=eval_comments, tpr=tpr,
    #     plot=plot,
    #     **kwargs)
    precision, recall, f1, average_precision, auroc, acc, df_results, threshold = eval_model(
        best_model, val_loader, test_loader, testing_datasets, class_label, eval_comments=eval_comments,
        quantile=quantile, tpr=tpr, pos_label=pos_label, normal_is_positive=normal_is_positive, plot=plot,
        plot_threshold=False, plot_name=model_name,
        **kwargs)

    return best_model, best_model_ckpt_path, precision, recall, f1, average_precision, auroc, acc, df_results, threshold


def eval_model(model, val_loader, test_loader, testing_datasets, class_label, eval_comments=False,
               quantile=0.05, tpr=False, pos_label=1, normal_is_positive=True, plot=True, device='cuda', plot_name=None,
               **kwargs):
    plot_threshold = kwargs.get("plot_threshold", plot)
    plot_val = kwargs.get("plot_val", plot)
    plot_overall = kwargs.get("plot_overall", plot)
    plot_metrics_per_class = kwargs.get("plot_metrics_per_class", plot)
    if device is None:
        torch_model = False
    else:
        torch_model = True
        model.to(device)
        model.eval()

    # val data
    if val_loader:
        print("Validation Data")
        threshold = get_threshold(
            model, val_loader, quantile=quantile, tpr=tpr, plot=plot_threshold, pos_label=pos_label,
            label_is_map=False, torch_model=torch_model)
        print("Threshold", threshold)
        _, _, _, _, _, _ = calculate_overall_metrics(
            model, val_loader, threshold, pos_label=pos_label, normal_is_positive=normal_is_positive,
            label_is_map=False, plot=plot_val, torch_model=torch_model, plot_name=plot_name)
    else:
        threshold = kwargs["threshold"]

    # test data
    print("Test Data")
    print("Overall Metrics")
    precision, recall, f1, average_precision, auroc, acc = calculate_overall_metrics(
        model, test_loader, threshold, pos_label=pos_label, normal_is_positive=normal_is_positive,
        eval_comments=eval_comments, plot=plot_overall, torch_model=torch_model, plot_name=plot_name)
    print(precision, recall, f1, average_precision, auroc, acc)

    # Metrics per Class
    if testing_datasets is not None:
        if len(testing_datasets) <= 2:
            # just normal and 1 anomaly type
            df_results = None
        else:
            print("Metrics per Class")
            df_results = metrics_per_class(
                model, testing_datasets, class_label, threshold, pos_label=pos_label,
                normal_is_positive=normal_is_positive,
                eval_comments=eval_comments, plot=plot_metrics_per_class, torch_model=torch_model, plot_name=plot_name)
            for col in df_results.columns:
                print(col, df_results[col].values)
            # print('precision', df_results['precision'].values)
            # print('recall', df_results['recall'].values)
            # print('f1', df_results['f1'].values)
            # print('average_precision', df_results['average_precision'].values)
            # print('auroc', df_results['auroc'].values)
    else:
        df_results = None

    return precision, recall, f1, average_precision, auroc, acc, df_results, threshold


def predict_from_loader(model, loader, torch_model, pos_label, label_is_map=False):
    """

    Args:
        model:
        loader: torch dataloader or (x, y) tuple of arrays
        torch_model:
        pos_label:
        label_is_map: bool if y label is anom label (0 for normal, 1/2/3/... for anom types)

    Returns:

    """
    if torch_model:
        y_score = []
        y_true = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        # Predict probabilities for each sample in the validation set
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs.to(device))
                y_score.extend(outputs.cpu().numpy())
                y_true.extend(targets.cpu().numpy())
        y_score = np.array(y_score)
        y_true = np.array(y_true)
    else:
        x, y_true = loader
        y_score = model.decision_function(x)
    if label_is_map:
        # convert base class (class 0) to 1 and others/anomalies (class 1/2/3/...) to 0
        y_true = (y_true == 0)
    if pos_label == 0:
        y_score = 1. - y_score  # for models w y_score btw -inf and inf, add 1 to make things easy
        y_true = 1 - y_true

    return y_score, y_true


def get_threshold(model, loader, quantile=0.90, tpr=True, pos_label=1, plot=True, label_is_map=False,
                  torch_model=True):
    if quantile is None:
        if torch_model:
            return 0.5
        else:
            return 1.

    y_score, y_true = predict_from_loader(
        model, loader, torch_model=torch_model, pos_label=pos_label, label_is_map=label_is_map)

    # Compute the false positive rate (FPR) and true positive rate (TPR) for different thresholds
    fpr_values, tpr_values, thresholds = roc_curve(y_true, y_score, pos_label=1)
    if plot:
        roc_auc = auc(fpr_values, tpr_values)
        display = RocCurveDisplay(fpr=fpr_values, tpr=tpr_values, roc_auc=roc_auc,
                                  estimator_name='')
        display.plot()
        plt.title("ROC Curve for Validation Data")
        plt.show()
        # plt.plot(thresholds, tpr_values)
        # plt.show()

    # Find the threshold that achieves the desired TPR
    if tpr:
        pos_preds = y_score[y_true == 1]
        threshold = np.quantile(pos_preds, quantile, method="closest_observation")
    # Find the threshold that achieves the desired FPR
    else:
        neg_preds = y_score[y_true == 0]
        threshold = np.quantile(neg_preds, 1 - quantile, method="closest_observation")

    return threshold


def calculate_overall_metrics(model, loader, threshold=0.5, pos_label=1, normal_is_positive=True, label_is_map=False,
                              eval_comments=False, plot=False, torch_model=True, plot_name=None):
    """

    Args:
        model:
        loader:
        threshold:
        pos_label:
        normal_is_positive: True if normal is positive class. else, False
        label_is_map: bool if y label is anom label (0 for normal, 1/2/3/... for anom types)
        eval_comments:
        plot:

    Returns:

    """

    y_scores, y_true = predict_from_loader(
        model, loader, torch_model=torch_model, pos_label=pos_label, label_is_map=label_is_map)

    # Calculate average precision
    average_precision = average_precision_score(y_true, y_scores, pos_label=1)
    #     average_precision = metric.compute()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_scores >= threshold, average='binary',
                                                               pos_label=1)
    acc = accuracy_score(y_true, y_scores >= threshold)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)

    if plot:
        idxes = np.where(thresholds == threshold)[0]
        if len(idxes) == 0:
            idx = np.argmax(thresholds >= threshold)
        else:
            idx = idxes[0]
        fpr_quantile = fpr[idx]
        if normal_is_positive:
            classes_plot = ["Normal", "Anomaly"]
        else:
            classes_plot = ["Anomaly", "Normal"]
        _, _ = plot_metrics(y_true, y_scores, classes=classes_plot, threshold=threshold,
                            recall=recall, fpr=fpr_quantile, eval_comments=eval_comments, plot_name=plot_name)

    return precision, recall, f1, average_precision, auroc, acc


def metrics_per_class(model, testing_datasets, class_label, threshold=0.5, pos_label=1, normal_is_positive=True,
                      label_is_map=False, eval_comments=False, plot=False, torch_model=True, plot_name=None):

    if torch_model:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_loader = torch.utils.data.DataLoader(testing_datasets[0], batch_size=1024, shuffle=False)
    else:
        test_loader = testing_datasets[0]

    pos_pred, pos_true = predict_from_loader(
        model, test_loader, torch_model=torch_model, pos_label=pos_label, label_is_map=label_is_map)

    # fake = "" if not flip_labels else " (NRF)"
    classes = {class_id: name for name, class_id in class_label.items()}

    df = pd.DataFrame(columns=["Class", "precision", "recall", "f1", "average_precision", "auroc", "acc"])
    for i in range(1, len(classes.keys())):
        if torch_model:
            test_loader = torch.utils.data.DataLoader(testing_datasets[i], batch_size=1024, shuffle=False)
        else:
            test_loader = testing_datasets[i]

        neg_pred, neg_true = predict_from_loader(
            model, test_loader, torch_model=torch_model, pos_label=pos_label, label_is_map=label_is_map)

        binary_true = np.hstack((pos_true, neg_true))
        binary_scores = np.hstack((pos_pred, neg_pred))

        precision, recall, f1, _ = precision_recall_fscore_support(binary_true, binary_scores >= threshold,
                                                                   average='binary', pos_label=1)
        acc = accuracy_score(binary_true, binary_scores >= threshold)

        if plot:
            if normal_is_positive:
                classes_plot = [classes[0], classes[i]]
            else:
                classes_plot = [classes[i], classes[0]]
            average_precision, roc_auc = plot_metrics(
                binary_true, binary_scores, classes=classes_plot, threshold=threshold,
                recall=recall, eval_comments=eval_comments, plot_name=plot_name)
        else:
            average_precision = average_precision_score(binary_true, binary_scores, pos_label=1)
            fprs, tprs, thresholds = roc_curve(binary_true, binary_scores)
            roc_auc = auc(fprs, tprs)

        print(classes[i], precision, recall, f1, average_precision, roc_auc, acc)
        df.loc[i - 1] = [classes[i], precision, recall, f1, average_precision, roc_auc, acc]
    return df


def plot_metrics(y_true, y_scores, classes, threshold=None, recall=None, eval_comments=False, fpr=None,
                 plot_name="OCC"):
    if plot_name is None:
        plot_name = "OCC"
    class0 = classes[0]
    class1 = classes[1]

    if eval_comments:
        append = eval_comments
        if eval_comments == " (NRF)":
            class0 = f"'{class0}'"
            class1 = f"'{class1}'"
    else:
        append = ""
    # Prediction Histogram
    pos_pred = y_scores[y_true == 1]
    neg_pred = y_scores[y_true == 0]

    plt.title(f"Prediction Histogram for {class0} vs {class1}{append} ({plot_name})")
    plt.hist(pos_pred, label=class0, color='red', alpha=0.7)
    plt.axvline(np.min(pos_pred), linestyle='--', color='red', alpha=0.4, label='Min Positive')
    plt.hist(neg_pred, label=class1, color='blue', alpha=0.7)
    plt.axvline(np.max(neg_pred), linestyle='--', color='blue', alpha=0.4, label='Max Negative')
    if threshold is not None:
        plt.axvline(x=threshold, color='orange', linestyle='--', alpha=0.9, label="Threshold")
    plt.legend()
    plt.show()

    # calculate precision and recall for each threshold
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
    pr_auc = auc(lr_recall, lr_precision)

    # plot the precision-recall curves
    no_skill = len(pos_pred) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    plt.plot(lr_recall, lr_precision, marker='.', label=f'{plot_name} (area = %0.3f)' % pr_auc)
    if recall is not None:
        plt.axvline(x=recall, color='orange', linestyle='--', alpha=0.8, label="Detection")

    # axis labels
    plt.title(f"PR-Curve{append}: {class0} vs {class1} ({plot_name})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend()
    plt.show()

    # Compute fpr, tpr, thresholds and roc auc
    fprs, tprs, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fprs, tprs)

    if fpr is not None:

        # Plot ROC curve
        plt.plot(fprs, tprs, label=f'{plot_name} (area = %0.3f)' % roc_auc)
        plt.axvline(fpr, color='red', linestyle='--', alpha=0.8)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve ({plot_name})')
        plt.legend(loc="lower right")
        plt.show()

    return pr_auc, roc_auc
