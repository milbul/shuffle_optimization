import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter


def results_to_file(directory_name: str, shuffle_size: list, jobs: list, variation_coef: list, mode: str):
    results_path = os.path.join('./results', directory_name)
    os.makedirs(results_path, exist_ok=True)
    write_list_to_file(os.path.join(results_path, 'jobs.txt'), jobs)
    write_list_to_file(os.path.join(results_path, f'{mode}_shuffle_size.txt'), shuffle_size)
    write_list_to_file(os.path.join(results_path, f'{mode}_variation_coefficient.txt'), variation_coef)


def write_list_to_file(path: str, data: list, delimiter: str = '\n', end: str = '\n'):
    with open(path, 'w') as file:
        file.write(delimiter.join(map(str, data)) + end)


def save_full_dataframe(directory_name: str, table: pd.DataFrame):
    path = os.path.join('./results', directory_name)
    os.makedirs(path, exist_ok=True)
    table.to_parquet(os.path.join(path, 'full_dataframe.parquet'))


def params_to_file(directory_name: str, nodes_number: int, allow_skew: float, first_group_by: list):
    path = os.path.join('./results', directory_name)
    os.makedirs(path, exist_ok=True)
    params = f"nodes_number = {nodes_number}\nallow_skew = {allow_skew}\nfirst_group_by = {first_group_by}"
    with open(os.path.join(path, 'params.txt'), 'w') as file:
        file.write(params)


def read_shuffle_size(path, table_size):
    shuffle_size = open(path).read().splitlines()
    shuffle_size = list(map(int, shuffle_size))
    shuffle_size = [x * 100 / table_size for x in shuffle_size]
    return shuffle_size


def compare_shuffle_size(path, table_size):
    results_path = os.path.join('./results', path)

    shuffle_files = {
        'Opt shuffle': 'optimized_shuffle_size.txt',
        'Hash shuffle': 'hash_shuffle_size.txt',
        'Holistic shuffle': 'holistic_shuffle_size.txt'
    }

    data = {label: read_shuffle_size(os.path.join(results_path, filename), table_size)
            for label, filename in shuffle_files.items()}

    df = pd.DataFrame(data)

    colors = ['#C09BFF', '#B3DDFF', '#FF7B75', '#B0D8AC']

    plt.style.use('dark_background')
    ax = df.plot(kind='bar', color=colors)

    ax.set_ylim(0, 100)
    ax.set_xlabel('Номер задания')
    ax.set_ylabel('% shuffle')

    ax2 = ax.twinx()
    ax2.set_ylim(0, table_size)
    ax2.set_ylabel('shuffle-данные')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((6, 6))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(6, 6))

    plot_path = os.path.join('./plots', f'{path}_compare_shuffle_size.png')
    plt.savefig(plot_path, transparent=True)


def read_variation_coefficient(path):
    variation_coefficient = open(path).read().splitlines()
    variation_coefficient = list(map(float, variation_coefficient))
    return variation_coefficient


def compare_variation_coefficient(path):
    results_path = os.path.join('./results', path)

    variation_coefficient_files = {
        'Opt shuffle': 'optimized_variation_coefficient.txt',
        'Hash shuffle': 'hash_variation_coefficient.txt',
        'Holistic shuffle': 'holistic_variation_coefficient.txt'
    }

    data = {label: read_variation_coefficient(os.path.join(results_path, filename))
            for label, filename in variation_coefficient_files.items()}

    df = pd.DataFrame(data)

    colors = ['#C09BFF', '#B3DDFF', '#FF7B75', '#B0D8AC']
    plt.style.use('dark_background')
    df.plot(kind='bar', color=colors)
    plt.ylim(0, 3)
    plt.xlabel('Номер задания')
    plt.ylabel('Коэффициэнт вариации')
    plot_path = os.path.join('./plots', f'{path}_compare_variation_coefficient.png')
    plt.savefig(plot_path, transparent=True)
