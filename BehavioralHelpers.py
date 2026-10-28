import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, 'dependencies/bidesreader')
from bidsreader import CMLBIDSReader as BIDSReader
sys.path.insert(0, ".")
from cml_data import (get_bids_root, available_subjects, available_tasks, session_dataframe)

def get_presented_words(subject, session, task):
    bids_root = get_bids_root(task, subject=subject, session=session)
    reader = BIDSReader(bids_root, task = task, subject = subject, session = session)
    evs = reader.load_events()
    presented_words = evs[evs['trial_type'] == 'WORD'] 
    return presented_words

def load_events(subject, session, task):
    bids_root = get_bids_root(task, subject=subject, session=session)
    reader = BIDSReader(bids_root, task = task, subject = subject, session = session)
    evs = reader.load_events()
    return evs

def get_recalled_words(subject, session, task):
    bids_root = get_bids_root(task, subject=subject, session=session)
    reader = BIDSReader(bids_root, task = task, subject = subject, session = session)
    evs = reader.load_events()
    recalled_words = evs[evs['trial_type'] == 'REC_WORD'] 
    return recalled_words

def recalled_column(presented_words, recalled_words):
    if 'list' in presented_words.columns and presented_words['list'].count()>0:
        list_col = 'list'
    elif 'trial' in presented_words.columns:
        list_col = 'trial'
    presented_words['recalled'] = 0
    for word, list in presented_words[['item_name', list_col]].values:
        if word in recalled_words[recalled_words[list_col] == list]['item_name'].values:
            presented_words.loc[(presented_words['item_name'] == word) & (presented_words[list_col] == list), 'recalled'] = 1
        else:
            presented_words.loc[(presented_words['item_name'] == word) & (presented_words[list_col] == list), 'recalled'] = 0
    return presented_words

def serialpos_column(presented_words):
    if 'list' in presented_words.columns and presented_words['list'].count()>0:
            list_col = 'list'
    elif 'trial' in presented_words.columns:
        list_col = 'trial'
    presented_words['serialpos'] = presented_words.groupby(list_col).cumcount() + 1
    return presented_words

def add_distractor_columns(evs):
    words = evs[evs['trial_type'] == 'WORD']
    last_words = words.groupby('trial').tail(1)
    lists = evs.dropna(subset=['trial'])

    distractor_by_trial = {}
    final_distractor_by_trial = {}
    for row in last_words.itertuples():
        evs_temp = lists[lists['trial'] == row.trial].reset_index(drop=True)
        evs_index = evs_temp.index[evs_temp['mstime'] == row.mstime][0]

        distractor = row.mstime - evs_temp['mstime'].iloc[evs_index-1]
        final_distractor = evs_temp['mstime'].iloc[evs_index+1] - row.mstime

        if 3000 <= final_distractor <= 8000:
            final_distractor_by_trial[row.trial] = 0
        elif final_distractor > 16000:
            final_distractor_by_trial[row.trial] = 16000
        else:
            final_distractor_by_trial[row.trial] = np.nan
        distractor_by_trial[row.trial] = 0 if distractor < 8000 else np.nan

    evs['distractor'] = evs['trial'].map(distractor_by_trial)
    evs['final_distractor'] = evs['trial'].map(final_distractor_by_trial)
    return evs
