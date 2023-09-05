import json

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.load.dump import dumps
from llama import Llama

prompt_template = PromptTemplate.from_template(
    """
    You are an advanced reasoning agent that can improve based on self reflection. You will be
    given a previous reasoning trial in which you were given access to an Docstore API environment
    and a question to answer. You were unsuccessful in answering the question either because you
    guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning
    steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise,
    high level plan that aims to mitigate the same failure. Use complete sentences.\n

    Here are some examples: 
    {few_shot_demonstation}\n

    Previous trial:
    {previous_trial}\n

    Reflection:
    """
)

class Retro:
    def __init__(self,
                 ckpt_dir,
                 tokenizer_path,
                 temperature,
                 top_p,
                 max_seq_len,
                 max_gen_len,
                 max_batch_size):
        self.replay_buffer = [] # Triplet (reflection prompt X_{k,i}, response: y_{k,i}, Return:G_{k,i}), trial i task k
        self.retro_temperature = temperature
        # here we trick langchain into thinking we are using openai model while we are actually using the fastchat one
        # conditioned on the fact that the server is in fact runing

        self.base_api_url = "http://localhost:8000"
        self.openai_api_key = "EMPTY"
        self.model_name = "gpt-3.5-turbo"
        self.model = ChatOpenAI(openai_api_base=self.base_api_url,
                                openai_api_key=self.openai_api_key,
                                model=self.model_name,
                                temperature=self.retro_temperature)

    def format_prompt(self, trajectory) -> str:
        """ Format the reflection prompt for each trial """
        print(json.dumps(trajectory, indent=0))

        return prompt_template.format(
            few_shot_demonstation="",
            previous_trial=trajectory
        )

    def get_reflections(self, trajectories, rewards):
        # loop over trajectories
        for trajectory, reward in zip(trajectories, rewards):
            trail_id = trajectory[0]
            question = trajectory[1]["input"]
            intermediate_steps = trajectory[1]["intermediate_steps"]
            final_answer = trajectory[1]["output"]
            correct_answer = trajectory[2]

            print("trail_id", trail_id)
            print("question", question)
            print("intermediate_steps", dumps(intermediate_steps, pretty=True))
            print("\nfinal_answer", final_answer)
            print("correct", correct_answer)
            print("rewards", rewards)
            print("\n\n\n")

            # format the reflection prompt for each trail
            # reflection_prompt = prompt_template.format(
            #     few_shot_demonstation="",
            #     previous_trial=output
            # )

    def backward_pass(self, trajectory):
        
        pass
            
