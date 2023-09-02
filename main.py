import os

import joblib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from actor import Actor
from ppo import PPOTrainer
from retro import Retro


def setup(rank, world_size):
    # Setup for distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the distributed environment
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Call the main function of your script
rank = 0
world_size = 1
mp.spawn(setup, args=(world_size), nprocs=world_size, join=True)

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

# Instantiate the retroformer model
retroformer = Retro(
    ckpt_dir="./llama/llama-2-7b",
    tokenizer_path="./llama/tokenizer.model",
    temperature=0,
    top_p=0.9,
    max_seq_len=512,
    max_gen_len=64,
    max_batch_size=6
)

number_of_tasks = 10 # amount of tasks (hotpotQA questions)
number_of_trails = 1 # amount of policy fine tunes per task

ppo = PPOTrainer(
                 n_tasks=number_of_tasks,
                 n_trails=number_of_trails,
                 actor_model=ACTOR_MODEL,
                 actor_model_temp=ACTOR_MODEL_TEMP,
                 retroformer=retroformer,
                 hotpotqa=hotpot
                )
ppo.train()