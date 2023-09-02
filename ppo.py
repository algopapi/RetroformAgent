from actor import Actor
from utils import normalize_answer


class PPOTrainer:
    def __init__(self, actor_model, actor_model_temp, retroformer, hotpotqa, n_trails, n_tasks):
        self.actor_model = actor_model # Actor
        self.actor_model_temp = actor_model_temp # Actor temperature
        self.retroformer = retroformer # Retrospective model
        self.hotpotqa = hotpotqa
        self.n_trails = n_trails # Number of tasks to solve
        self.n_tasks = n_tasks # number of agents per trail
        self.selected_tasks = self.hotpotqa[:self.n_tasks]
        print("selected tasks", self.selected_tasks)
        self.agents = [Actor(trail_id, task, self.actor_model, self.actor_model_temp)
               for trail_id, task in self.selected_tasks.iterrows()]
        
        self.rewards = [] # G_{k,i} for each trail i and task k
       

    def f1_score(self, reference, candidate):
        """
        Calculate the F1 score between a reference answer and a candidate answer.
        
        Args:
        - reference (str): the reference (or gold standard) answer
        - candidate (str): the model-generated answer
        
        Returns:
        - float: the F1 score between the two answers
        """
        
        # Tokenize the answers into words
        reference_tokens = set(reference.split())
        candidate_tokens = set(candidate.split())
        
        # Calculate the number of shared tokens between the two answers
        common = reference_tokens.intersection(candidate_tokens)
        
        # If there are no shared tokens, the F1 score is 0
        if not common:
            return 0.0
    
        # Calculate precision and recall
        precision = len(common) / len(candidate_tokens)
        recall = len(common) / len(reference_tokens)
        
        # Calculate the F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

    def get_tasks(self):
        """get 'batch_size' hotpot questions from hotpotqa"""
        hotpot_sample = self.hotpotqa.sample(self.n_trails)

        print(" hotpot sample", hotpot_sample)

    def gather_trajectories(self) -> list:
        """ gathers answers from K agents under the current policy"""
        answers = [agent.rollout() for agent in self.agents]
        return answers

    def get_rewards(self, trajectories) -> list:
        """ get the returns from the trajectories"""
        rewards = []
        for trail_id, output, answer in trajectories:
            final_answer = output['output']
            f1_score = self.f1_score(normalize_answer(answer), normalize_answer(final_answer))
            rewards.append(f1_score)
        return rewards
    
    def get_reflections(self, trajectories) -> list[str]:
        """get the reflections from the trajectories"""
        for trail_id, output, answer in trajectories:
            pass

    def train(self):
        for trail in range(self.n_trails):
            # Gather N_trails trajectories under current policy
            trajectories = self.gather_trajectories()
            print("gathered trajectories:", trajectories)

            # Get the rewards from the answers from current trail
            self.rewards.append(self.get_rewards(trajectories))
            print("self.rewards", self.rewards)
            print("\ncurrent:",  self.rewards[trail])
            
            #reflections = self.get_reflections(trajectories)


            # Reflection rating = G_{k,i+1} - G{k, i}
            reflection_ratings = self.rewards[trail] - self.rewards[trail - 1]
            print("reflection ratings:", reflection_ratings)



            # Fine tune the policy on gathered trajectories.
            