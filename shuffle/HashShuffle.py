import numpy as np
import pandas as pd

from results_to_file import results_to_file


class HashShuffle:
    def __init__(self, df: pd.DataFrame, jobs: list, default_partition_keys: list, nodes_number: int, directory_name: str = None):
        self.df = df
        self.jobs = jobs
        self.default_partition_keys = default_partition_keys
        self.nodes_number = nodes_number
        self.directory_name = directory_name
        self.hash_shuffle = []
        self.variation_coef = []
        self.pipeline = [job.replace(" ", "").split(',') for job in jobs]

    def repartition(self, group_by_columns: list, prev_group_by_columns: list) -> int:
        if set(prev_group_by_columns).issubset(set(group_by_columns)):
            return 0

        self.df = self.df.rename(columns={'node': 'prev_node'})
        self.df['hash'] = pd.util.hash_pandas_object(self.df[group_by_columns], index=False, categorize=True)
        self.df['new_node'] = self.df['hash'] % self.nodes_number + 1

        shuffle_size = self.df.loc[self.df['prev_node'] != self.df['new_node']].shape[0]
        self.df = self.df.rename(columns={'new_node': 'node'}).drop(columns=['prev_node', 'hash'])
        return shuffle_size

    def first_partition(self):
        self.repartition(self.default_partition_keys, self.df.columns.tolist())

    def variation_coefficient(self, values: pd.Series) -> float:
        empty_nodes = [0] * (self.nodes_number - len(values))
        values.extend(empty_nodes)
        c = np.std(values) / np.mean(values)
        return np.std(values) / np.mean(values)

    def get_node_distribution(self):
        return self.df.groupby('node')['node'].count().reset_index(name="count").sort_values('count', ascending=False)

    def shuffle(self):
        self.df['node'] = 1
        self.first_partition()

        prev_columns = self.default_partition_keys
        for columns in self.pipeline:
            shuffle_size = self.repartition(columns, prev_columns)
            self.hash_shuffle.append(shuffle_size)
            node_distribution = self.get_node_distribution()
            sizes = node_distribution['count'].tolist()
            self.variation_coef.append(self.variation_coefficient(sizes))
            prev_columns = columns

        if self.directory_name is not None:
            results_to_file(self.directory_name, self.hash_shuffle, self.jobs, self.variation_coef, "hash")
