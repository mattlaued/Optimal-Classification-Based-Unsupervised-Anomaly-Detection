import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

import re


def collate_results(
        results, result_cols=["precision", "recall", "f1", "average_precision", "auroc", "acc", "df_results"]):
    """

    Args:
        results: list of metric-wise results
        result_cols: metrics

    Returns:

    """
    d = dict()
    for col in result_cols:
        d[col] = list()
    for result in results:
        for col, col_name in zip(result, result_cols):
            if col is None:
                continue
            d[col_name].append(col)

    return d


def get_mean_df_results(list_of_df_results, key="df_results"):
    if key == "df_results":
        df_overall = pd.concat([df.drop(["Class"], axis="columns") for df in list_of_df_results]).groupby(level=0)
        mean = df_overall.mean()
        std = df_overall.std()
        classes = list_of_df_results[0]["Class"]
        mean.insert(loc=0, column="Class", value=classes)
        std.insert(loc=0, column="Class", value=classes)
    else:
        list_of_df_results = np.array(list_of_df_results)
        mean = np.mean(list_of_df_results)
        std = np.std(list_of_df_results)
    return mean, std


def agg_results(results):
    aggregated_results = dict()
    for k, v in results.items():
        if len(v) == 0:
            aggregated_results[k] = []
        else:
            mean, std = get_mean_df_results(list_of_df_results=v, key=k)
            # if k == "df_results":
            #     df_overall = pd.concat([df.drop(["Class"], axis="columns") for df in v]).groupby(level=0)
            #     mean = df_overall.mean()
            #     std = df_overall.std()
            #     classes = v[0]["Class"]
            #     mean.insert(loc=0, column="Class", value=classes)
            #     std.insert(loc=0, column="Class", value=classes)
            # else:
            #     v = np.array(v)
            #     mean = np.mean(v)
            #     std = np.std(v)
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


def print_exp_agg_df_results(df_results):
    print("Metrics by Class:", list(df_results[0]["Class"]))
    print(f"Precision:", print_mean_std(zip(df_results[0]["precision"], df_results[1]["precision"])))
    print(f"Recall:", print_mean_std(zip(df_results[0]["recall"], df_results[1]["recall"])))
    print(f"F1:", print_mean_std(zip(df_results[0]["f1"], df_results[1]["f1"])))
    print(f"Average Precision:", print_mean_std(zip(df_results[0]["average_precision"],
                                                    df_results[1]["average_precision"])))
    print(f"AUROC:", print_mean_std(zip(df_results[0]["auroc"], df_results[1]["auroc"])))
    print(f"Accuracy:", print_mean_std(zip(df_results[0]["acc"], df_results[1]["acc"])))


def print_exp_agg_results(precision, recall, f1, average_precision, auroc, acc, df_results):

    if precision is None:
        # No overall results. Just print per-class results
        print_exp_agg_df_results(df_results)
    else:
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
                print_exp_agg_df_results(df_results)
                # print("Metrics by Class:", list(df_results[0]["Class"]))
                # print(f"Precision:", print_mean_std(zip(df_results[0]["precision"], df_results[1]["precision"])))
                # print(f"Recall:", print_mean_std(zip(df_results[0]["recall"], df_results[1]["recall"])))
                # print(f"F1:", print_mean_std(zip(df_results[0]["f1"], df_results[1]["f1"])))
                # print(f"Average Precision:", print_mean_std(zip(df_results[0]["average_precision"],
                #                                                 df_results[1]["average_precision"])))
                # print(f"AUROC:", print_mean_std(zip(df_results[0]["auroc"], df_results[1]["auroc"])))
                # print(f"Accuracy:", print_mean_std(zip(df_results[0]["acc"], df_results[1]["acc"])))
            except:
                print("No Anomaly Type Information")


def print_mean_std(ls):
    s = ""
    for mean, std in ls:
        s += f"{mean}$\pm${std} & "
    return s[:-2] + "\\\\"


def round_results(results, delimiter="&", pm="$\pm$", round_off=3, save="str", num_results_to_use=1):
    if save == "str":
        s = ""
    else:
        s = []
    units = results.split(delimiter)
    for i, unit in enumerate(units):
        mean, std = unit.strip().split(pm)
        mean, std = round(float(mean), round_off), round(float(std), round_off)
        if save == "str":
            s += f"{str(mean).ljust(5,'0')}$\pm${str(std).ljust(5,'0')} & "
        else:
            s.append((mean, std))
        if i+1 >= num_results_to_use:
            break

    if save == "str":
        return s[:-2] + "\\\\"
    else:
        return s


def extract_results_from_file(
        file_path, metric="Average Precision:", sf=3, num_results_to_use=1, metric_count=4, metric_number=0,
        baseline_metric=False):

    results = []

    with open(file_path, 'r') as file:
        result = ""
        metric_counter = 0
        while line := file.readline():
            content = line.strip()
            if content.startswith('\"cell_type\":'):
                # if result is in the next cell, save all results as 1 bunch and move on
                if result != "":
                    results.append(result[:-3])
                    result = ""
                    metric_counter = 0
            if content.startswith(f"\"{metric}"):
                metric_counter += 1
                if metric_counter % metric_count == metric_number:
                    stripped_content = content.strip(",\"").strip().strip("\\n").removeprefix(f"{metric}").strip()
                    if baseline_metric:
                        result += str(round(float(stripped_content), sf))
                    else:
                        result += round_results(
                            stripped_content, round_off=sf, num_results_to_use=num_results_to_use, pm="$\\\\pm$")[:-3]
                    result += " & "

        if result != "":
            results.append(result[:-3])

    for r in results:
        print(r)
    return results


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


def strip_comment(line):
    # Find all '%' characters not preceded by a backslash
    comment_match = re.search(r'(?<!\\)%', line)
    if comment_match:
        return line[:comment_match.start()].rstrip()
    else:
        return line


def remove_latex_comments(latex_code):
    """
    Removes LaTeX comments from the input string.
    A LaTeX comment starts with an unescaped '%' and continues to the end of the line.
    Escaped percent signs (i.e., '\%') are preserved.
    """
    # Process each line to remove comments
    return '\n'.join(strip_comment(line) for line in latex_code.splitlines())


def transpose_latex_table(latex_table):
    latex_table = remove_latex_comments(latex_table)
    # Split input string into individual rows
    rows = [line.strip() for line in latex_table.splitlines() if line.strip() and '\\midrule' not in line]

    # Create a matrix of table contents
    table_data = []
    for row in rows:
        # Remove \\ at the end and split by '&' to get each cell value
        cells = [cell.strip() for cell in re.split(r'\s*&\s*', row.rstrip('\\'))]
        table_data.append(cells)

    # Transpose the table
    transposed_table = list(map(list, zip(*table_data)))

    # Generate the transposed LaTeX table string
    transposed_latex = ''

    # Create the first row with the new headers
    transposed_latex += ' & '.join(transposed_table[0]) + '\\\\\n\\midrule\n'

    # Create the remaining rows with the new contents
    for row in transposed_table[1:]:
        transposed_latex += ' & '.join(row) + '\\\\\n'

    return transposed_latex


def swap_latex_columns(latex_table: str, col1: int, col2: int) -> str:
    lines = latex_table.splitlines()
    updated_lines = []
    for line in lines:
        # Ignore formatting lines like \midrule, \toprule, etc.
        if re.match(r'\\(mid|top|bottom)rule', line):
            updated_lines.append(line)
            continue
        # Remove comments from the end of the line
        line, _, comment = line.partition('%')
        # Ensure proper handling of LaTeX newlines
        line = line.rstrip()
        has_newline = line.endswith('\\')
        line = line[:-2].strip() if has_newline else line.strip()
        # Split columns by '&' while preserving LaTeX commands
        columns = [col.strip() for col in line.split('&')]
        if len(columns) > max(col1, col2):  # Ensure swap is valid
            columns[col1], columns[col2] = columns[col2], columns[col1]
            updated_line = ' & '.join(columns)
            if has_newline:
                updated_line += ' \\\\'
        else:
            updated_line = line  # Keep unmodified if not a data row
        # Add back the comment if it existed
        if comment:
            updated_line += ' %' + comment
        updated_lines.append(updated_line)
    return '\n'.join(updated_lines)


def get_metric_rows(s, metric="Average Precision: ", overall=True, num_per_dataset=2, round_off=3, file=True):
    if file:
        # then s is the path of the results file to be read. jupyter notebook is used here
        with open(s, "r") as f:
            s = f.read()
    relevant_lines = []
    lines = s.split("\n")
    for line in lines:
        if metric in line:
            l = line.strip().strip(",").strip('\"').replace("\\\\", "\\").replace("\\n", "").replace(metric, "").strip()
            relevant_lines.append(l)
            # print(l)
        # elif "kwargs_data" in line:
        #     print("***********************************")
    r = int(not overall)
    results = [list() for _ in range(num_per_dataset)]
    for i, line in enumerate(relevant_lines):
        j = i - r
        if j % 2 == 0:
            if j % 4 != 0:
                continue
            num = (j // 4) % num_per_dataset
            # if round_off:
            #     line = round_results(line, round_off=round_off)
            results[num].append(line)
            # for num in range(num_per_dataset):
            #     if (j // 4) %   * (num + 1)) == 0:
            #         results[num].append(line)
            #         break
    for result in results:
        print("****************************")
        s = " & ".join(result)
        if round_off:
            s = round_results(s, round_off=round_off)
        print(s)
        # for line in result:
        #     print(line)


def latex_to_markdown(latex_table: str) -> str:
    # Extract rows from LaTeX table
    rows = [re.split(r'(?<!\\)%', line, 1)[0].strip() for line in latex_table.splitlines() if line.strip() and not line.startswith('%')]
    r = []
    for row in rows:
        if 'tabular' not in row and r'\hline' not in row and "rule" not in row:
            stripped = re.sub(r'\\\\', '', row)
            r.append(stripped)
    rows = r

    # rows = [re.sub(r'\\\\', '', row) for row in rows]  # Remove LaTeX row endings
    # rows = [row for row in rows if 'tabular' not in row and r'\hline' not in row and "rule" not in row]  # Remove tabular and hline lines

    # Extract column data
    table_data = []
    for row in rows:
        cols = re.split(r'&', row)
        c = []
        for col in cols:
            s = re.sub(r'\\textbf{(.*?)}', r'**\1**', col.strip())  # Convert \textbf to **...**
            s = s.replace('$\\pm$', '±')
            c.append(s)
        cols = c
        if cols:
            table_data.append(cols)

    # Ensure table is not empty and has uniform column count
    if not table_data:
        raise ValueError("Malformed table: No valid rows detected.")
    col_count = len(table_data[0])
    if any(len(row) != col_count for row in table_data):
        print(col_count)
        print([len(row) for row in table_data])
        print(table_data)
        raise ValueError("Malformed table: inconsistent column counts.")

    # Find max column widths
    col_widths = [max(len(row[i]) for row in table_data) for i in range(len(table_data[0]))]

    # Format Markdown table
    markdown_table = "|" + "|".join(f"{col}" for i, col in enumerate(table_data[0])) + "|\n"
    markdown_table += "|" + "|".join("-" for i in range(len(col_widths))) + "|\n"
    for row in table_data[1:]:
        markdown_table += "|" + "|".join(f"{col}" for i, col in enumerate(row)) + "|\n"

    return markdown_table.strip()


def sample_data(data, prop_data, seed):
    np.random.seed(seed)
    indices = np.random.choice(len(data), size=int(prop_data * len(data)), replace=False)
    return data[indices]


def viz_kdd(data_2d, y_train_real, x_train_synthetic_anom_feat, y_val_real, x_val_synthetic_anom_feat, y_test, viz_method="UMAP",
            normal_colour="blue", synthetic_anom_colour="green", known_anom_colour="black",
            unknown_anom_colour="red",
            marker_size=7, marker_size_known_anom=20, marker_train="x", marker_val="+",
            alpha=0.3, alpha_known_anom=0.9,
            prop_data_for_viz=0.01, prop_test_anoms_for_viz=0.05, use_synthetic_anoms=True, save=None):
    seed_viz = 321

    num_train_real = len(y_train_real)
    num_train_synthetic_anom = len(x_train_synthetic_anom_feat)
    num_val_real = len(y_val_real)
    num_val_synthetic_anom = len(x_val_synthetic_anom_feat)
    num_test_real = len(y_test)

    start_index = 0
    end_index = num_train_real
    known_anom_indices = (y_train_real == 0)
    normal_indices = (y_train_real == 1)
    plt.scatter(data_2d[start_index:end_index, 0][known_anom_indices],
                data_2d[start_index:end_index, 1][known_anom_indices],
                c=known_anom_colour, alpha=alpha_known_anom, s=marker_size_known_anom, marker=marker_train,
                label="Known Anom")
    # pick fewer normal data for viz
    normal_data = data_2d[start_index:end_index][normal_indices]
    # np.random.seed(seed_viz)
    # normal_indices = np.random.choice(len(normal_data), size=int(prop_data_for_viz*len(normal_data)), replace=False)
    normal_data = sample_data(normal_data, prop_data_for_viz, seed_viz)
    ###################################
    plt.scatter(normal_data[:, 0], normal_data[:, 1],
                c=normal_colour, alpha=alpha, s=marker_size, marker=marker_train, label="Normal")
    # plt.scatter(data_2d[start_index:end_index, 0][normal_indices], data_2d[start_index:end_index, 1][normal_indices], 
    #             c=normal_colour, alpha=alpha, s=marker_size)

    if use_synthetic_anoms and num_train_synthetic_anom > 0:
        start_index = end_index
        end_index += num_train_synthetic_anom
        # pick fewer synthetic data for viz
        synthetic_anoms = data_2d[start_index:end_index]
        # np.random.seed(seed_viz)
        # synthetic_indices = np.random.choice(len(synthetic_anoms), size=int(prop_data_for_viz*len(synthetic_anoms)), replace=False)
        # synthetic_anoms = synthetic_anoms[synthetic_indices]
        synthetic_anoms = sample_data(synthetic_anoms, prop_data_for_viz, seed_viz)
        ###################################
        plt.scatter(synthetic_anoms[:, 0], synthetic_anoms[:, 1],
                    c=synthetic_anom_colour, alpha=alpha, s=marker_size, marker=marker_train, label="Syn. Anom")
        # plt.scatter(data_2d[start_index:end_index, 0], data_2d[start_index:end_index, 1],
        #             c=synthetic_anom_colour, alpha=alpha, s=marker_size)
    # else:
    #     use_synthetic_anoms = False

    start_index = end_index
    end_index += num_val_real
    known_anom_indices = (y_val_real == 0)
    normal_indices = (y_val_real == 1)
    plt.scatter(data_2d[start_index:end_index, 0][known_anom_indices],
                data_2d[start_index:end_index, 1][known_anom_indices],
                c=known_anom_colour, alpha=alpha_known_anom, s=marker_size_known_anom, marker=marker_val,
                label="Known Anom")
    # pick fewer normal data for viz
    normal_data = data_2d[start_index:end_index][normal_indices]
    # np.random.seed(seed_viz)
    # normal_indices = np.random.choice(len(normal_data), size=int(prop_data_for_viz*len(normal_data)), replace=False)
    # normal_data = normal_data[normal_indices]
    normal_data = sample_data(normal_data, prop_data_for_viz, seed_viz)
    ###################################
    plt.scatter(normal_data[:, 0], normal_data[:, 1],
                c=normal_colour, alpha=alpha, s=marker_size, marker=marker_val, label="Normal")
    # plt.scatter(data_2d[start_index:end_index, 0][normal_indices], data_2d[start_index:end_index, 1][normal_indices], 
    #             c=normal_colour, alpha=alpha, s=marker_size)

    if use_synthetic_anoms and num_val_synthetic_anom > 0:
        start_index = end_index
        end_index += num_val_synthetic_anom
        # pick fewer synthetic data for viz
        synthetic_anoms = data_2d[start_index:end_index]
        # np.random.seed(seed_viz)
        # synthetic_indices = np.random.choice(len(synthetic_anoms), size=int(prop_data_for_viz*len(synthetic_anoms)), replace=False)
        # synthetic_anoms = synthetic_anoms[synthetic_indices]
        synthetic_anoms = sample_data(synthetic_anoms, prop_data_for_viz, seed_viz)
        ###################################
        plt.scatter(synthetic_anoms[:, 0], synthetic_anoms[:, 1],
                    c=synthetic_anom_colour, alpha=alpha, s=marker_size, marker=marker_val, label="Syn. Anom")
        # plt.scatter(data_2d[start_index:end_index, 0], data_2d[start_index:end_index, 1],
        #             c=synthetic_anom_colour, alpha=alpha, s=marker_size)

    start_index = end_index
    end_index += num_test_real
    normal_indices = (y_test == 0)
    known_anom_indices = (y_test == 3)
    unknown_anom_indices = np.isin(y_test, [1, 2, 4])
    # pick fewer normal data for viz
    normal_data = data_2d[start_index:end_index][normal_indices]
    np.random.seed(seed_viz)
    normal_indices = np.random.choice(len(normal_data), size=int(prop_data_for_viz * len(normal_data)), replace=False)
    normal_data = normal_data[normal_indices]
    ###################################
    plt.scatter(normal_data[:, 0], normal_data[:, 1],
                c=normal_colour, alpha=alpha, s=marker_size, label="Normal")
    # plt.scatter(data_2d[start_index:end_index, 0][normal_indices], data_2d[start_index:end_index, 1][normal_indices], 
    #             c=normal_colour, alpha=alpha, s=marker_size)
    plt.scatter(data_2d[start_index:end_index, 0][known_anom_indices],
                data_2d[start_index:end_index, 1][known_anom_indices],
                c=known_anom_colour, alpha=alpha_known_anom, s=marker_size_known_anom, label="Known Anom")

    # pick fewer unknown anom for viz
    # unknown_anom_data = data_2d[start_index:end_index][unknown_anom_indices]
    dos_indices = (y_test == 1)
    dos_data = data_2d[start_index:end_index][dos_indices]
    dos_data = sample_data(dos_data, prop_test_anoms_for_viz, seed_viz)
    probe_indices = (y_test == 2)
    probe_data = data_2d[start_index:end_index][probe_indices]
    probe_data = sample_data(probe_data, prop_test_anoms_for_viz, seed_viz)
    priv_esc_indices = (y_test == 4)
    priv_esc_data = data_2d[start_index:end_index][priv_esc_indices]
    # priv_esc_data = sample_data(priv_esc_data, prop_data_for_viz, seed_viz)

    plt.scatter(dos_data[:, 0], dos_data[:, 1],
                c=unknown_anom_colour, alpha=alpha, s=marker_size, marker="s", label="Unk. Anom (DoS)")
    plt.scatter(probe_data[:, 0], probe_data[:, 1],
                c=unknown_anom_colour, alpha=alpha, s=marker_size, marker="*", label="Unk. Anom (Probe)")
    plt.scatter(priv_esc_data[:, 0], priv_esc_data[:, 1],
                c=unknown_anom_colour, alpha=alpha, s=marker_size, marker="H", label="Unk. Anom (Priv)")

    # plt.scatter(data_2d[start_index:end_index, 0][unknown_anom_indices], data_2d[start_index:end_index, 1][unknown_anom_indices], 
    #             c=unknown_anom_colour, alpha=alpha, s=marker_size, label="Unknown Anom")

    print("All Data Accounted for:", end_index == len(data_2d))

    normal_patch = mpatches.Patch(color=normal_colour, label='Normal')
    known_anom_patch = mpatches.Patch(color=known_anom_colour, label='Known Anom')
    unknown_anom_patch = mpatches.Patch(color=unknown_anom_colour, label='Unknown Anom')
    colours = [normal_patch, known_anom_patch, unknown_anom_patch]
    if use_synthetic_anoms:
        syn_anom_patch = mpatches.Patch(color=synthetic_anom_colour, label='Syn. Anom')
        colours.append(syn_anom_patch)

    train_datum = Line2D([0], [0], label='Train Datum', marker=marker_train, markersize=5, linestyle='')
    val_datum = Line2D([0], [0], label='Val Datum', marker=marker_val, markersize=5, linestyle='')
    # test_datum = Line2D([0], [0], label='Test Datum', marker="o", markersize=5, linestyle='')
    dos_datum = Line2D([0], [0], label='Unk: DoS', marker="s", markersize=5, linestyle='')
    probe_datum = Line2D([0], [0], label='Unk: Probe', marker="*", markersize=5, linestyle='')
    priv_esc_datum = Line2D([0], [0], label='Unk: Priv.', marker="p", markersize=5, linestyle='')
    data_points = [train_datum, val_datum
                   #      , test_datum
        , dos_datum, probe_datum, priv_esc_datum
                   ]

    legend_handles = data_points + colours

    legend = plt.legend(handles=legend_handles, ncol=2,
                          loc="lower left",
               bbox_to_anchor=(-0.15, -0.15)
              )
    legend.get_frame().set_alpha(0.9)

    plt.xlabel(f'{viz_method} Component 1')
    # Access the current Axes object
    ax = plt.gca()

    # Adjust the position of the x-axis label
    # For example, move it slightly above its default position
    ax.xaxis.set_label_coords(0.65, -0.095)  # Adjust the y-coordinate as needed
    plt.ylabel(f'{viz_method} Component 2')
    title = f'{viz_method} of Model Predictions (NSL-KDD)'
    if use_synthetic_anoms:
        title += " with Syn. Anoms"
    else:
        title += " w/o Syn. Anoms"
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

