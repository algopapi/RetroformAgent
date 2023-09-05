import os

import joblib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from actor import Actor
from ppo import PPOTrainer
from retro import Retro

# Prepare the hotpotQA data#
hotpot = joblib.load('./data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    supporting_articles = row['supporting_facts']['title']
    articles = row['context']['title']
    sentences = row['context']['sentences']
    supporting_paragraphs = []
    for article in supporting_articles:
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs

# create an instance of the ppo trainer
ACTOR_MODEL = "gpt-4"
ACTOR_MODEL_TEMP = 0
with_context = True

# Instantiate the retroformer model
retroformer = Retro(
    temperature=0,
    with_context=with_context
)

number_of_tasks = 2 # amount of tasks (hotpotQA questions)
number_of_trials = 1 # amount of policy fine tunes per task

f1_threshold = 0.7

ppo = PPOTrainer(n_tasks=number_of_tasks,
                 n_trials=number_of_trials,
                 actor_model=ACTOR_MODEL,
                 actor_model_temp=ACTOR_MODEL_TEMP,
                 retroformer=retroformer,
                 hotpotqa=hotpot,
                 with_context=with_context,
                 f1_threshold=f1_threshold)
ppo.train()