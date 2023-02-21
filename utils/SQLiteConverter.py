# use the example from https://www.kaggle.com/code/rasmusrse/graphnet-example
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import sqlite3
import sqlalchemy
import sys
import os

from tqdm import tqdm
from typing import Any, Dict, List, Optional
from graphnet.data.sqlite.sqlite_utilities import create_table
'''
I am not sure right now
import pandas as pd
import sqlite3

my_event_id = 32
my_database = '/kaggle/working/sqlite/batch_01.db'

with sqlite3.connect(my_database) as conn:
    # extracts meta data for event
    meta_query = f'SELECT * FROM meta_table WHERE event_id ={my_event_id}'
    meta_data = pd.read_sql(meta_query,conn)

    # extracts pulses / detector response for event
    pulse_query = f'SELECT * FROM pulse_table WHERE event_id ={my_event_id}'
    pulse_data = pd.read_sql(query,conn)
'''

# auxiliary 这里应该要删除掉True的我觉得
def load_input(meta_batch: pd.DataFrame, input_data_folder: str) -> pd.DataFrame:
    """
    Will load the corresponding detector readings associated with the meta data batch.
    """
    batch_id = pd.unique(meta_batch['batch_id'])

    assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"

    detector_readings = pd.read_parquet(path=f'{input_data_folder}/batch_{batch_id[0]}.parquet')
    sensor_positions = geometry_table.loc[detector_readings['sensor_id'], ['x', 'y', 'z']]
    sensor_positions.index = detector_readings.index

    for column in sensor_positions.columns:
        if column not in detector_readings.columns:
            detector_readings[column] = sensor_positions[column]

    detector_readings['auxiliary'] = detector_readings['auxiliary'].replace({True: 1, False: 0})
    return detector_readings.reset_index()

def add_to_table(database_path: str,
                      df: pd.DataFrame,
                      table_name:  str,
                      is_primary_key: bool,
                      ) -> None:
    """Writes meta data to sqlite table.

    Args:
        database_path (str): the path to the database file.
        df (pd.DataFrame): the dataframe that is being written to table.
        table_name (str, optional): The name of the meta table. Defaults to 'meta_table'.
        is_primary_key(bool): Must be True if each row of df corresponds to a unique event_id. Defaults to False.
    """
    try:
        create_table(   columns=  df.columns,
                        database_path = database_path,
                        table_name = table_name,
                        integer_primary_key= is_primary_key,
                        index_column = 'event_id')
    except sqlite3.OperationalError as e:
        if 'already exists' in str(e):
            pass
        else:
            raise e
    engine = sqlalchemy.create_engine("sqlite:///" + database_path)
    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize = 200000)
    engine.dispose()
    return
# batch_size  200000 是因为一个batch正好200000条event 我感觉我们先用200000条测试一下速度就行了
def convert_to_sqlite(meta_data_path: str,
                      database_path: str,
                      input_data_folder: str,
                      batch_size: int = 200000,
                      batch_ids: Optional[List[int]] = None,) -> None:
    """Converts a selection of the Competition's parquet files to a single sqlite database.

    Args:
        meta_data_path (str): Path to the meta data file.
        batch_size (int): the number of rows extracted from meta data file at a time. Keep low for memory efficiency.
        database_path (str): path to database. E.g. '/my_folder/data/my_new_database.db'
        input_data_folder (str): folder containing the parquet input files.
        batch_ids (List[int]): The batch_ids you want converted. Defaults to None (all batches will be converted)
    """
    if batch_ids is None:
        batch_ids = np.arange(1,661,1).to_list()
    else:
        assert isinstance(batch_ids,list), "Variable 'batch_ids' must be list."
    if not database_path.endswith('.db'):
        database_path = database_path+'.db'
    meta_data_iter = pq.ParquetFile(meta_data_path).iter_batches(batch_size = batch_size)
    batch_id = 1
    converted_batches = []
    progress_bar = tqdm(total = len(batch_ids))
    for meta_data_batch in meta_data_iter:
        if batch_id in batch_ids:
            meta_data_batch  = meta_data_batch.to_pandas()
            add_to_table(database_path = database_path,
                        df = meta_data_batch,
                        table_name='meta_table',
                        is_primary_key= True)
            pulses = load_input(meta_batch=meta_data_batch, input_data_folder= input_data_folder)
            del meta_data_batch # memory
            add_to_table(database_path = database_path,
                        df = pulses,
                        table_name='pulse_table',
                        is_primary_key= False)
            del pulses # memory
            progress_bar.update(1)
            converted_batches.append(batch_id)
        batch_id +=1
        if len(batch_ids) == len(converted_batches):
            break
    progress_bar.close()
    del meta_data_iter # memory
    print(f'Conversion Complete!. Database available at\n {database_path}')


input_data_folder = '/home/yicong/icecube-neutrinos-in-deep-ice/train'
geometry_table = pd.read_csv('/home/yicong/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')
meta_data_path = '/home/yicong/icecube-neutrinos-in-deep-ice/train_meta.parquet'
database_path = '/home/yicong/working/batch_1.db'
convert_to_sqlite(meta_data_path,
                  database_path=database_path,
                  input_data_folder=input_data_folder,
                  batch_ids = [1])