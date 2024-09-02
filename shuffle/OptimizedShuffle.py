import math

import pandas as pd
import numpy as np
import logging

from results_to_file import results_to_file, save_full_dataframe


class OptimizedShuffle:
    def __init__(self, df: pd.DataFrame, jobs: list, default_partition_keys: list, nodes_number: int, allow_skew: float,
                 mode: str, directory_name: str = None):
        self.df = df
        self.columns = df.columns.tolist()
        self.jobs = jobs
        self.pipeline = [sorted(job.replace(" ", "").split(',')) for job in jobs]
        self.default_partition_keys = default_partition_keys
        self.nodes_number = nodes_number
        self.directory_name = directory_name
        self.allow_skew = allow_skew
        self.mode = mode
        self.optimized_shuffle = []
        self.variation_coef = []
        self.full_result = None

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

    def repartition(self, table: pd.DataFrame, partition_columns: list, prev_partition_columns: list) \
            -> (pd.DataFrame, int):
        if set(prev_partition_columns).issubset(set(partition_columns)):
            return table, 0

        partition_cols = partition_columns + ['node']
        nodes_stats = table.copy().groupby(partition_cols)['node'].count().reset_index(name="count")
        nodes_stats['count'] = nodes_stats['count'].fillna(value=0)
        nodes_stats['count_all'] = nodes_stats.groupby(partition_columns)['count'].transform('sum')
        ideal_size = np.ceil(table.shape[0] / self.nodes_number)

        if nodes_stats['count'].equals(nodes_stats['count_all']):
            current_partition = table.groupby('node')['node'].count().reset_index(name="count")
            if not current_partition[current_partition['count'] > ideal_size * self.allow_skew].empty:
                return table, 0

        nodes_stats['new_node'] = -1
        columns_node_stats = nodes_stats.columns.tolist()
        rows = nodes_stats

        nodes_sizes = [0] * self.nodes_number
        result = table.copy().rename(columns={'node': 'prev_node'})
        while not rows.empty:
            parts = rows.groupby(partition_columns)['count'].idxmax()
            rows = rows.loc[parts].sort_values(['count'], ascending=False)
            rows['count_cumsum'] = rows.groupby('node')['count_all'].cumsum()
            rows['lag_count_cumsum'] = rows.groupby('node')['count_cumsum'].shift(1)
            max_part = rows.loc[rows['lag_count_cumsum'] > self.allow_skew * ideal_size]
            if not max_part.empty:
                partition_cols = max_part.iloc[0]['node']
                rows['new_node'] = np.where(((rows['lag_count_cumsum'] < self.allow_skew * ideal_size)
                                             | (rows['lag_count_cumsum'].isnull()))
                                            & (rows['node'] == partition_cols)
                                            & (~rows['count'].isnull()), rows['node'], -1)
            else:
                rows['new_node'] = rows['node'].astype('int')

            rows = rows.loc[rows['new_node'] != -1]
            c = rows.loc[rows.groupby(['new_node'])["count_cumsum"].idxmax()]
            for _, row in c.iterrows():
                nodes_sizes[int(row['new_node']) - 1] = row['count_cumsum']

            nodes_stats = pd.merge(nodes_stats, rows, on=partition_columns, how='left', suffixes=('', '_1'))
            nodes_stats['new_node'] = np.where(
                (nodes_stats['new_node'].notnull()) & (nodes_stats['new_node'] != -1),
                nodes_stats['new_node'],
                nodes_stats['new_node_1'])
            nodes_stats['new_node'] = nodes_stats['new_node'].fillna(-1).astype('int')
            nodes_stats = nodes_stats[columns_node_stats]

            used_nodes = [i + 1 for i, x in enumerate(nodes_sizes) if x > 0]
            rows = nodes_stats.loc[(~nodes_stats['node'].isin(used_nodes)) & (nodes_stats['new_node'] == -1)]

        while not nodes_stats.loc[nodes_stats['new_node'] == -1].empty:
            node_minimal = nodes_sizes.index(min(nodes_sizes))
            max_rows_count = int(self.allow_skew * ideal_size - nodes_sizes[node_minimal])

            rows = nodes_stats.loc[nodes_stats['new_node'] == -1][partition_columns + ['node', 'new_node']]
            rows = rows.groupby(partition_columns)['new_node'].count().reset_index(name="count")
            rows = rows.sort_values('count', ascending=False)
            rows['count_cumsum'] = rows['count'].cumsum()
            rows['lag_count_cumsum'] = rows['count_cumsum'].shift(1)
            rows = rows.loc[(rows['lag_count_cumsum'].isnull()) | (rows['lag_count_cumsum'] < max_rows_count)]
            rows['new_node'] = node_minimal + 1
            rows = rows.loc[rows['new_node'] != -1]
            nodes_sizes[node_minimal] += rows['count_cumsum'].max()

            nodes_stats = pd.merge(nodes_stats, rows, on=partition_columns, how='left', suffixes=('', '_1'))
            nodes_stats['new_node'] = np.where(
                (nodes_stats['new_node'].notnull()) & (nodes_stats['new_node'] != -1),
                nodes_stats['new_node'],
                nodes_stats['new_node_1'])
            nodes_stats['new_node'] = nodes_stats['new_node'].fillna(-1).astype('int')
            nodes_stats = nodes_stats[partition_columns + ['node', 'new_node']]
        nodes_stats = nodes_stats[partition_columns + ['new_node']].drop_duplicates()
        result = pd.merge(result, nodes_stats, on=partition_columns, how='inner')
        shuffle_size = result[result['prev_node'] != result['new_node']].shape[0]
        result = result.rename(columns={'new_node': 'node'}).drop(columns=['prev_node'])
        return result, shuffle_size

    def first_partition_hash(self) -> int:
        self.df.rename(columns={'node': 'prev_node'}, inplace=True)
        self.df['hash'] = pd.util.hash_pandas_object(self.df[self.default_partition_keys], index=False, categorize=True)
        self.df['new_node'] = self.df['hash'] % self.nodes_number + 1
        shuffle_size = self.df.loc[self.df['prev_node'] != self.df['new_node']].shape[0]
        self.df = self.df.rename(columns={'new_node': 'node'}).drop(['prev_node', 'hash'], axis=1)
        return shuffle_size

    def variation_coefficient(self, values: pd.Series) -> float:
        empty_nodes = [0] * (self.nodes_number - len(values))
        values.extend(empty_nodes)
        return np.std(values) / np.mean(values)

    def shuffle_short_pipeline(self, first_task: int, last_task: int, prev_columns: [],
                               similarity_pipeline: [[]] = None, compare_shuffle: int = None):
        shuffle_sizes = []
        full_shuffle_size = 0
        cur_df = self.df.copy()
        prev_partition = prev_columns
        variation_coef = []
        if self.mode == 'F':
            full_result = self.full_result.copy()
        for i in range(first_task, last_task):
            job = similarity_pipeline[i] if similarity_pipeline else self.pipeline[i]
            cur_df, shuffle_size = self.repartition(cur_df.copy(), job, prev_partition)
            prev_partition = job

            full_shuffle_size += shuffle_size
            shuffle_sizes.append(shuffle_size)

            node_counts = cur_df.groupby('node')['node'].count().reset_index(name="count") \
                .sort_values('count', ascending=False)
            sizes = node_counts['count'].tolist()
            variation_coef.append(self.variation_coefficient(sizes.copy()))
            if self.mode == 'F':
                full_result = pd.merge(full_result, cur_df, on=self.columns, how='inner')
                full_result.rename(columns={'node': f'node_{i}'}, inplace=True)

            if compare_shuffle is not None and full_shuffle_size > compare_shuffle:
                return full_shuffle_size, shuffle_sizes, cur_df, variation_coef, full_result

        return full_shuffle_size, shuffle_sizes, cur_df, variation_coef, full_result

    def _update_pipeline_results(self, shuffle_sizes, variation_coef, full_result):
        self.optimized_shuffle.extend(shuffle_sizes)
        self.variation_coef.extend(variation_coef)
        if self.mode == 'F':
            self.full_result = full_result

    def check_optimized_pipeline(self, first_task: int, last_task: int, similarity_pipeline: [[]],
                                 prev_columns: []) -> pd.DataFrame:

        opt_full_shuffle_size, opt_shuffle_sizes, opt_cur_df, opt_variation_coef, opt_full_result = \
            self.shuffle_short_pipeline(first_task, last_task, prev_columns.copy(), similarity_pipeline.copy())

        def_full_shuffle_size, def_shuffle_sizes, def_cur_df, def_variation_coef, def_full_result = \
            self.shuffle_short_pipeline(first_task, last_task, prev_columns.copy(),
                                        compare_shuffle=opt_full_shuffle_size)
        if opt_full_shuffle_size < def_full_shuffle_size:
            self._update_pipeline_results(opt_shuffle_sizes, opt_variation_coef, opt_full_result)
            return opt_cur_df

        self._update_pipeline_results(def_shuffle_sizes, def_variation_coef, def_full_result)
        return def_cur_df

    def find_similarity_job_keys(self, unique_count: pd.Series) -> [[]]:
        n = len(self.pipeline)
        optimized_pipeline = [[] for _ in range(n)]
        cur_job = 0

        while cur_job < n:
            cur_intersect = self.pipeline[cur_job]
            next_job = cur_job + 1

            while len(cur_intersect) > 0 and next_job < n:
                intersect = list(set(cur_intersect).intersection(self.pipeline[next_job]))

                if not intersect:
                    break

                cur_columns = unique_count[intersect]
                multiple_value_columns = cur_columns[cur_columns > 1]
                if ((cur_columns > self.nodes_number * 10).any()) \
                        or math.log(self.nodes_number * 10) + 1 < len(multiple_value_columns) \
                        or math.prod(cur_columns.to_list()) > self.nodes_number * 10:
                    cur_intersect = intersect
                else:
                    break
                next_job += 1
            for job in range(cur_job, next_job):
                optimized_pipeline[job] = sorted(cur_intersect)

            cur_job = next_job
        return optimized_pipeline

    def get_node_distribution(self):
        return self.df.groupby('node')['node'].count().reset_index(name="count")

    def find_last_similar_job(self, current_job, similarity_pipeline):
        last_similar_job = current_job + 1
        while last_similar_job < len(self.pipeline) and \
                similarity_pipeline[last_similar_job] == similarity_pipeline[current_job]:
            last_similar_job += 1
        return last_similar_job

    def repartition_optimized_keys(self, current_job, last_similar_job, similarity_pipeline, prev_columns):
        self.df = self.check_optimized_pipeline(current_job, last_similar_job, similarity_pipeline.copy(),
                                                prev_columns.copy())

    def repartition_default_keys(self, job, similarity_job, prev_columns, current_job):
        self.df, shuffle_size = self.repartition(self.df.copy(), similarity_job.copy(), prev_columns.copy())
        if self.mode == 'F':
            self.full_result = pd.merge(self.full_result, self.df, on=self.columns, how='inner')
            self.full_result.rename(columns={'node': f'node_{current_job}'}, inplace=True)

        self.optimized_shuffle.append(shuffle_size)
        node_distribution = self.get_node_distribution()
        sizes = node_distribution['count'].tolist()
        self.variation_coef.append(self.variation_coefficient(sizes.copy()))

    def save_results(self):
        if self.directory_name is not None:
            results_to_file(self.directory_name, self.optimized_shuffle, self.jobs, self.variation_coef, "optimized")
            if self.mode == 'F':
                save_full_dataframe(self.directory_name, self.full_result)

    def shuffle(self):
        logging.info('Collecting table statistics.')
        used_columns = list(set().union(*self.pipeline))
        unique_count = self.df[used_columns].nunique()
        similarity_pipeline = self.find_similarity_job_keys(unique_count)
        self.df['node'] = 1
        logging.info('Determining default nodes')
        self.first_partition_hash()

        prev_columns = self.default_partition_keys.copy()
        current_job = 0
        if self.mode == 'F':
            self.full_result = self.df.copy().rename(columns={'node': 'node_default'})
        while current_job < len(self.pipeline):
            similarity_job = similarity_pipeline[current_job]
            job = self.pipeline[current_job]
            logging.info(f'Current job is {job}.')
            if job != similarity_job:
                last_similar_job = self.find_last_similar_job(current_job, similarity_pipeline)
                self.repartition_optimized_keys(current_job, last_similar_job, similarity_pipeline, prev_columns)
                current_job = last_similar_job
            else:
                self.repartition_default_keys(job, similarity_job, prev_columns, current_job)
                current_job += 1
            prev_columns = similarity_job

        self.save_results()
