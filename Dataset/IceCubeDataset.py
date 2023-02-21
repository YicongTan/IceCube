from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sqlite3
import tqdm
def make_selection(df: pd.DataFrame, pulse_threshold: int = 200) -> None:
    """Creates a validation and training selection (20 - 80). All events in both selections satisfies n_pulses <= 200 by default. """
    n_events = np.arange(0, len(df), 1)
    train_selection, validate_selection = train_test_split(n_events,
                                                           shuffle=True,
                                                           random_state=42,
                                                           test_size=0.20)
    df['train'] = 0
    df['validate'] = 0

    df['train'][train_selection] = 1
    df['validate'][validate_selection] = 1

    assert len(train_selection) == sum(df['train'])
    assert len(validate_selection) == sum(df['validate'])

    # Remove events with large pulses from training and validation sample (memory)
    df['train'][df['n_pulses'] > pulse_threshold] = 0
    df['validate'][df['n_pulses'] > pulse_threshold] = 0

    for selection in ['train', 'validate']:
        df.loc[df[selection] == 1, :].to_csv(f'{selection}_selection_max_{pulse_threshold}_pulses.csv')
    return


def get_number_of_pulses(db: str, event_id: int, pulsemap: str) -> int:
    with sqlite3.connect(db) as con:
        query = f'select event_id from {pulsemap} where event_id = {event_id} limit 20000'
        data = con.execute(query).fetchall()
    return len(data)


def count_pulses(database: str, pulsemap: str) -> pd.DataFrame:
    """ Will count the number of pulses in each event and return a single dataframe that contains counts for each event_id."""
    with sqlite3.connect(database) as con:
        query = 'select event_id from meta_table'
        events = pd.read_sql(query, con)
    counts = {'event_id': [],
              'n_pulses': []}
    for event_id in tqdm.tqdm(events['event_id']):
        a = get_number_of_pulses(database, event_id, pulsemap)
        counts['event_id'].append(event_id)
        counts['n_pulses'].append(a)
    df = pd.DataFrame(counts)
    df.to_csv('counts.csv')
    return df

pulsemap = 'pulse_table'
database = '/home/yicong/working/batch_1.db'

df = count_pulses(database, pulsemap)
make_selection(df = df, pulse_threshold =  200)


import matplotlib.pyplot as plt
import pandas as pd
fig = plt.figure(figsize=(6,4), constrained_layout = True)
plt.hist(df['n_pulses'], histtype = 'step', label = 'batch_1', bins = np.arange(0,400,1))
plt.xlabel('# of Pulses', size = 15);
plt.xticks(size = 15);
plt.yticks(size = 15);
plt.plot(np.repeat(200,2), [0, 4000], label = f'Selection\n{np.round((sum(df["n_pulses"]<= 200)/len(df))*100, 1)} % pass' )
plt.legend(frameon = False, fontsize = 15);