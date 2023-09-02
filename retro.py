import json

from langchain import PromptTemplate

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

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def format_prompt(self, trajectory) -> str:
        """ Format the reflection prompt for each trial """
        print(json.dumps(trajectory, indent=0))

        return prompt_template.format(
            few_shot_demonstation="",
            previous_trial=trajectory
        )

    def foward_pass(self, trajectories):
        # loop over trajectories
        for trail_id, output, answer in trajectories:
            # format the reflection prompt for each trail
            reflection_prompt = prompt_template.format(
                few_shot_demonstation="",
                previous_trial=output
            )
            
