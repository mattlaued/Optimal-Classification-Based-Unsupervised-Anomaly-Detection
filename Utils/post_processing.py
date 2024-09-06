import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def collate_results(
        results, result_cols=["precision", "recall", "f1", "average_precision", "auroc", "acc", "df_results"]):
    d = dict()
    for col in result_cols:
        d[col] = list()
    for result in results:
        for col, col_name in zip(result, result_cols):
            if col is None:
                continue
            d[col_name].append(col)

    return d


def agg_results(results):
    aggregated_results = dict()
    for k, v in results.items():
        if len(v) == 0:
            aggregated_results[k] = []
        else:
            if k == "df_results":
                df_overall = pd.concat([df.drop(["Class"], axis="columns") for df in v]).groupby(level=0)
                mean = df_overall.mean()
                std = df_overall.std()
                classes = v[0]["Class"]
                mean.insert(loc=0, column="Class", value=classes)
                std.insert(loc=0, column="Class", value=classes)
            else:
                v = np.array(v)
                mean = np.mean(v)
                std = np.std(v)
            aggregated_results[k] = (mean, std)

    return aggregated_results


def agg_results_diff_datasets(list_of_agg_results):
    """
    get and print metrics across different anomaly datasets
    Args:
        list_of_agg_results: list of aggregated results

    Returns: df results of different anomaly datasets (mean, std)

    """
    df_results_mean = pd.concat([agg["df_results"][0].iloc[0] for agg in list_of_agg_results], axis=1).T
    df_results_std = pd.concat([agg["df_results"][1].iloc[0] for agg in list_of_agg_results], axis=1).T
    df_results = [df_results_mean, df_results_std]

    print("Metrics by Class:", list(df_results[0]["Class"]))
    print(f"Precision:", print_mean_std(zip(df_results[0]["precision"], df_results[1]["precision"])))
    print(f"Recall:", print_mean_std(zip(df_results[0]["recall"], df_results[1]["recall"])))
    print(f"F1:", print_mean_std(zip(df_results[0]["f1"], df_results[1]["f1"])))
    print(f"Average Precision:", print_mean_std(zip(df_results[0]["average_precision"],
                                                    df_results[1]["average_precision"])))
    print(f"AUROC:", print_mean_std(zip(df_results[0]["auroc"], df_results[1]["auroc"])))
    print(f"Accuracy:", print_mean_std(zip(df_results[0]["acc"], df_results[1]["acc"])))
    return df_results


def print_exp_agg_results(precision, recall, f1, average_precision, auroc, acc, df_results):

    if len(precision) == 0:
        print("NIL")
    else:
        print(f"Precision: {precision[0]}$\pm${precision[1]}")
        print(f"Recall: {recall[0]}$\pm${recall[1]}")
        print(f"F1: {f1[0]}$\pm${f1[1]}")
        print(f"Average Precision: {average_precision[0]}$\pm${average_precision[1]}")
        print(f"AUROC: {auroc[0]}$\pm${auroc[1]}")
        print(f"Accuracy: {acc[0]}$\pm${acc[1]}")
        try:
            print("Metrics by Class:", list(df_results[0]["Class"]))
            print(f"Precision:", print_mean_std(zip(df_results[0]["precision"], df_results[1]["precision"])))
            print(f"Recall:", print_mean_std(zip(df_results[0]["recall"], df_results[1]["recall"])))
            print(f"F1:", print_mean_std(zip(df_results[0]["f1"], df_results[1]["f1"])))
            print(f"Average Precision:", print_mean_std(zip(df_results[0]["average_precision"],
                                                            df_results[1]["average_precision"])))
            print(f"AUROC:", print_mean_std(zip(df_results[0]["auroc"], df_results[1]["auroc"])))
            print(f"Accuracy:", print_mean_std(zip(df_results[0]["acc"], df_results[1]["acc"])))
        except:
            print("No Anomaly Type Information")


def print_mean_std(ls):
    s = ""
    for mean, std in ls:
        s += f"{mean}$\pm${std} & "
    return s[:-2] + "\\\\"


def round_results(results, delimiter="&", pm="$\pm$", round_off=3, save="str"):
    if save == "str":
        s = ""
    else:
        s = []
    units = results.split(delimiter)
    for unit in units:
        mean, std = unit.strip().split(pm)
        mean, std = round(float(mean), round_off), round(float(std), round_off)
        if save == "str":
            s += f"{str(mean).ljust(5,'0')}$\pm${str(std).ljust(5,'0')} & "
        else:
            s.append((mean, std))

    if save == "str":
        return s[:-2] + "\\\\"
    else:
        return s


def colour_ablations(table, cols=[2, 3, 4, 5], inc="blue", dec="red"):
    lines = table.split("\n")
    # remove commented out lines
    lines = [line.strip().removesuffix("\\\\").strip() for line in lines if line.strip()[0] != "%"]
    original = lines[0].split("&")
    vals = [original[i].strip() for i in cols]
    means = []
    stds = []
    for val in vals:
        mean, std = val.split("$\\pm$")
        means.append(float(mean))
        stds.append(float(std))
    for line in lines[1:]:
        if "rule" in line:
            print(line)
            continue
        if "olor" in line:
            print(line + " \\\\")
            continue
        cells = line.split("&")
        print_line = ""
        curr_col = 0
        for i, cell in enumerate(cells):
            cell = cell.strip()
            if i in cols:
                # compare
                mean, std = cell.split("$\\pm$")
                original_mean = means[curr_col]
                original_std = stds[curr_col]
                diff = float(mean) - original_mean
                if diff < -original_std:
                    print_line += "{\\color{" + dec + "}" + cell + "}"
                elif diff > original_std:
                    print_line += "{\\color{" + inc + "}" + cell + "}"
                else:
                    print_line += cell
                curr_col += 1
            else:
                print_line += cell
            print_line += " & "
        print(print_line[:-2] + "\\\\")


def grab_metrics_from_results(aggregated_results, metric_name='acc', anomaly_type_info=True):
    metrics_mean = []
    metrics_std = []
    for agg in aggregated_results:
        m = agg[metric_name]
        metrics_mean.append(m[0])
        metrics_std.append(m[1])

    if anomaly_type_info:
        anom_indiv_mean = []
        anom_indiv_std = []
        for agg in aggregated_results:
            anom_indiv_mean.append(agg['df_results'][0][metric_name].to_numpy().squeeze())
            anom_indiv_std.append(agg['df_results'][1][metric_name].to_numpy().squeeze())

        return metrics_mean, metrics_std, anom_indiv_mean, anom_indiv_std
    return metrics_mean, metrics_std


def plot_agg_metrics(agg_metrics, x_axis, x_axis_label, y_axis_label,
                     anom_type=["Denial of Service", "Probe", "Remote Access", "Privilege Escalation"],
                     x_scale=None, y_scale=None):
    if len(agg_metrics) > 2:
        overall = agg_metrics[:2]
        anom_indiv_mean, anom_indiv_std = agg_metrics[2:]
        plot_agg_metrics(
            overall, x_axis, x_axis_label, y_axis_label, anom_type="Overall", x_scale=x_scale, y_scale=y_scale)
        anom_indiv_mean = np.array(anom_indiv_mean).T
        anom_indiv_std = np.array(anom_indiv_std).T
        for cls, mean, std in zip(anom_type,anom_indiv_mean, anom_indiv_std):
            plot_agg_metrics(
                (mean, std), x_axis, x_axis_label, y_axis_label, anom_type=cls, x_scale=x_scale, y_scale=y_scale)

    else:
        mean, std = agg_metrics
        # plt.plot(x_axis, mean, )
        plt.errorbar(x_axis, mean, yerr=std, ecolor='black', capsize=3)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        if x_scale is not None:
            plt.xscale(x_scale)
        if y_scale is not None:
            plt.yscale(y_scale)
        plt.title(f"{y_axis_label} for Normal and {anom_type}")
        plt.show()


def get_percent_str(percent):
    if percent.is_integer():
        percent_str = f"{int(percent)}_0"
    else:
        percent_str = str(percent)
        if "e" in percent_str:
            nums = percent_str.split("-")
            num_zeros = int(nums[-1])
            percent_str = "0" * (num_zeros - 1) + nums[0][:-1][:2]
        else:
            percent_str = percent_str.split(".")[-1]
        percent_str = "0_" + percent_str

    return percent_str


def check_model_size(path, directory=True):
    if not directory:
        ckpt_dir = os.path.join(path, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])
        ckpt = torch.load(ckpt_path)
        dim = ckpt['state_dict']["classifier.0.weight"].size()[-1]
        print(path, "dim:", dim)
        return dim
    else:
        folders = os.listdir(path)
        num_folders = len(folders)
        dims = []
        for i in range(num_folders):
            folder = "version_{}".format(i)
            try:
                ckpt_dir = os.path.join(path, folder, "checkpoints")
                ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])
                ckpt = torch.load(ckpt_path)
                dim = ckpt['state_dict']["classifier.0.weight"].size()[-1]
            except Exception as e:
                dim = None
                print(e)
            print(folder, "dim:", dim)
            dims.append(dim)
        return dims
