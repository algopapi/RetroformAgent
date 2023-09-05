import pprint
from typing import Dict

from actor import Actor

pp = pprint.PrettyPrinter(indent=4)


class ReplayBuffer:
    def __init__(self):
        self.buffer = {}
    
    def add(self, k, i, x, y, G):
        """
        k: task_id
        i: trial_id
        x: reflection prompt
        y: reflection response
        G: reward 
        """
        if k not in self.buffer:
            self.buffer[k] = {}
        self.buffer[k][i] = (x, y, G)

    def get(self, k, i):
        return self.buffer[k][i]
    
    def __str__(self):
        return str(self.buffer)
    

class PPOTrainer:
    def __init__(self, actor_model, actor_model_temp, retroformer, hotpotqa, with_context, n_trials, n_tasks, f1_threshold):
        self.actor_model = actor_model # Actor
        self.actor_model_temp = actor_model_temp # Actor temperature
        self.retroformer = retroformer # Retrospective model
        self.hotpotqa = hotpotqa
        self.with_context = with_context
        self.n_trials = n_trials # Number of tasks to solve
        self.n_tasks = n_tasks # number of agents per trail
        self.selected_tasks = self.hotpotqa[:self.n_tasks]
        self.f1_threshold = 0.7
        self.agents = [Actor(task_id, task, self.with_context, self.actor_model, self.actor_model_temp)
               for task_id, task in self.selected_tasks.iterrows()]
        
        self.past_trajectories = {} #(trial, trajectory) 
        self.rewards = [] # (task_id, G_{k,i}) for each trail i and task k
        self.replay_buffer = ReplayBuffer() # Triplet (reflection prompt X_{k,i}, response: y_{k,i}, Return:G_{k,i}), trial i task k
       
    def get_tasks(self):
        """get 'batch_size' hotpot questions from hotpotqa"""
        hotpot_sample = self.hotpotqa.sample(self.n_trials)

        print("Hotpot sample", hotpot_sample)

    def gather_trajectories(self) -> Dict[int, Dict]:
        """ gathers answers from K agents under the current policy"""
        trajectories = [agent.rollout() for agent in self.agents]
        return {tr["task_id"]: {"response": tr["response"], "reflection_prompt": tr["reflection_prompt"], "f1_score": tr["f1_score"]} for tr in trajectories}


    def train(self):
        for trial in range(self.n_trials):
            # Gather N_trails trajectories under current policy
            trajectories = self.gather_trajectories() # dict (task_id, response, reflection_prompt, rewards)

            print("Trajectories")
            pp.pprint(trajectories)

            # Store trajectories
            self.past_trajectories[trial] = trajectories # store the trajectories for the current trial.
        
            # Get the reflection
            reflections = self.retroformer.generate_reflections(trajectories) #(task_id, reflection)

            # pass the reflection to the agent (with corresponding task_id)
            # and add the triplet to the replay buffer
            for agent in self.agents:
                agent_reflection = reflections[agent.task_id]
                agent.update_policy(agent_reflection)
                self.replay_buffer.add(agent.task_id,
                                       trial,
                                       trajectories[agent.task_id]["reflection_prompt"], # reflection prompt
                                       agent_reflection,
                                       trajectories[agent.task_id]["f1_score"]) # the reward
                

            #reflections = self.get_reflections(trajectories)
            #Reflection rating = G_{k,i+1} - G{k, i}
            #reflection_ratings = self.rewards[trail] - self.rewards[trail - 1]
            #print("reflection ratings:", reflection_ratings)

            # print reflections
            print("Reflections")
            print(reflections)
            

                

            # Fine tune!!
