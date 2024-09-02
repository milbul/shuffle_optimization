import numpy as np
import pandas as pd

from results_to_file import results_to_file


class HolisticShuffle:
    def __init__(self, df: pd.DataFrame, jobs: list, default_partition_keys: list, nodes_number: int,
                 directory_name: str = None):
        self.df = df
        self.jobs = jobs
        self.default_partition_keys = default_partition_keys
        self.nodes_number = nodes_number
        self.directory_name = directory_name
        self.holistic_shuffle = []
        self.variation_coef = []
        self.pipeline = [sorted(job.replace(" ", "").split(',')) for job in jobs]

    def repartition(self, partition_columns: list, prev_partition_columns: list) -> (pd.DataFrame, int):
        if set(prev_partition_columns).issubset(set(partition_columns)):
            return self.df, 0

        table = self.df.copy()
        group_cols = partition_columns + ['node']
        separated_nodes = table.groupby(group_cols)['node'].count().reset_index(name="count")
        separated_nodes['count_all'] = separated_nodes.groupby(partition_columns)['count'].transform('sum')
        separated_nodes = separated_nodes.sort_values('count', ascending=False)

        if separated_nodes['count'].equals(separated_nodes['count_all']):
            return table, 0

        rows = separated_nodes
        table = table.rename(columns={'node': 'prev_node'})
        table['new_node'] = -1
        result = table.copy()
        result_columns = list(table.columns)
        parts = rows.groupby(partition_columns)['count'].idxmax()
        rows = rows.loc[parts].sort_values(['count'], ascending=False)
        rows['new_node'] = rows['node']
        result = pd.merge(result, rows, on=partition_columns, how='left', suffixes=('_1', ''))

        result = result[result_columns]
        shuffle_size = result.loc[result['prev_node'] != result['new_node']].shape[0]

        result = result.rename(columns={'new_node': 'node'}).drop('prev_node', axis=1)
        return result, shuffle_size

    def first_partition_hash(self) -> int:
        self.df = self.df.rename(columns={'node': 'prev_node'})
        self.df['hash'] = pd.util.hash_pandas_object(self.df[self.default_partition_keys], index=False, categorize=True)
        self.df['node'] = self.df['hash'] % self.nodes_number + 1
        shuffle_size = self.df.loc[self.df['prev_node'] != self.df['node']].shape[0]
        self.df.drop(columns=['prev_node', 'hash'], inplace=True)
        return shuffle_size

    def variation_coefficient(self, values: pd.Series) -> float:
        empty_nodes = [0] * (self.nodes_number - len(values))
        values.extend(empty_nodes)
        return np.std(values) / np.mean(values)

    def get_node_distribution(self):
        return self.df.groupby('node')['node'].count().reset_index(name="count")

    def shuffle(self):
        self.df['node'] = 1
        self.first_partition_hash()

        prev_columns = self.default_partition_keys
        for columns in self.pipeline:
            self.df, shuffle_size = self.repartition(columns, prev_columns)
            self.holistic_shuffle.append(shuffle_size)
            node_distribution = self.get_node_distribution()
            sizes = node_distribution['count'].tolist()
            self.variation_coef.append(self.variation_coefficient(sizes.copy()))
            prev_columns = columns
        if self.directory_name is not None:
            results_to_file(self.directory_name, self.holistic_shuffle, self.jobs, self.variation_coef, "holistic")
