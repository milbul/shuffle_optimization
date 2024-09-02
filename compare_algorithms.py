import sys

import pandas as pd
import pyarrow.parquet as pq
import datetime
from datetime import datetime

from shuffle.HashShuffle import HashShuffle

from shuffle.HolisticShuffle import HolisticShuffle
from shuffle.OptimizedShuffle import OptimizedShuffle
from results_to_file import params_to_file, compare_variation_coefficient, compare_shuffle_size


def compare_shuffle(df: pd.DataFrame, jobs: list, default_partition_keys, nodes_number: int, allow_skew: float, mode):
    directory_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    hash_shuffle = HashShuffle(df.copy(), jobs.copy(), default_partition_keys.copy(), nodes_number, directory_name)
    hash_shuffle.shuffle()
    optimized_shuffle = OptimizedShuffle(df.copy(), jobs.copy(), default_partition_keys.copy(), nodes_number,
                                         allow_skew, mode, directory_name)
    optimized_shuffle.shuffle()
    holistic_shuffle = HolisticShuffle(df.copy(), jobs.copy(), default_partition_keys.copy(), nodes_number,
                                       directory_name)
    holistic_shuffle.shuffle()
    params_to_file(directory_name, nodes_number, allow_skew, default_partition_keys)
    compare_shuffle_size(path=directory_name, table_size=df.shape[0])
    compare_variation_coefficient(path=directory_name)


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    jobs = sys.argv[2].replace(" ", "").split(';')
    allow_skew = float(sys.argv[3])
    nodes_number = int(sys.argv[4])
    default_partition_keys = sys.argv[5].replace(" ", "").split(';')
    mode = sys.argv[6]
    df = pq.read_table(dataset_path).to_pandas()
    compare_shuffle(df, jobs, default_partition_keys, nodes_number, allow_skew, mode)
