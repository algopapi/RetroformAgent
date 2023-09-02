from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, Wikipedia
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX
from langchain.agents.react.base import DocstoreExplorer
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

load_dotenv(find_dotenv())

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
SUFFIX = """Begin!
Reflection History: {long_term_memory}
Current Reflection: {policy}
Relevant Context: {context}
Question: {input}
Thought:{agent_scratchpad}"""

class Actor:
  def __init__(self, trail_id, task, model="gpt-4", model_temperature=0):
    self.trail_id = trail_id
    self.task = task
    self.question = task["question"]
    self.context = task['supporting_paragraphs']
    self.answer = task["answer"]
    print(f"trail_id: {self.trail_id}\n Question: {self.question}\n\n Supporting Paragraph: {self.context}\n\n Answer: {self.answer}\n\n")

    self.episode = 0
    self.actor_model = ChatOpenAI(model=model, temperature=model_temperature) # define the actor model (frozen weights) (probably some gpt-4)
    self.retroformer_prompt = "" # Initialize with an empty policy prompt
    self.docstore = DocstoreExplorer(Wikipedia())
    self.tools = [
      Tool(
        name="Search",
        func=self.docstore.search,
        description="useful for when you need to ask with search"
      ),
      Tool(
        name="Lookup",
        func=self.docstore.lookup,
        description="useful for when you need to ask with lookup (strictly use after succesful search)"
      )
    ]

    self.prompt = ZeroShotAgent.create_prompt(
      self.tools,
      prefix=PREFIX,
      format_instructions=FORMAT_INSTRUCTIONS,
      suffix=SUFFIX,
      input_variables=["input", "agent_scratchpad", "context", "policy", "long_term_memory"]
    )

    self.llm_chain = LLMChain(llm=self.actor_model, prompt=self.prompt)
    self.tool_names = [tool.name for tool in self.tools]
    self.agent = ZeroShotAgent(llm_chain=self.llm_chain,
                               allowed_tools=self.tool_names,
                               max_iterations=10)
    
    self.short_term_memory = [] # trajectory history t_i of current episode i
    self.long_term_memory = [] # The reflection responses that summearize prior failed attemps

  def add_reflection_response(self, reflection):
    self.long_term_memory.append(reflection)
  
  def clear_reflection_response(self):
    self.long_term_memory =[]

  def format_longterm_memory(self) -> str:
    """ Generates a prompt string from the list of long term reflections from the agent
        [".1.", " .2. ", ".3."] ->
       output format:
        \n
        .1. \n
        .2. \n
        .3. \n
        \n
    """
    
    formatted_reflections = "\n".join([reflection.strip() for reflection in self.long_term_memory])
    return f"\n\n{formatted_reflections}\n\n"
  
  def _handle_error(self, error) -> str:
    return str(error)[:50]

  def rollout(self):
    """ Get a single agent answer under the current policy"""
    output = None
    agent_executor = AgentExecutor.from_agent_and_tools(
      agent=self.agent,
      tools=self.tools,
      verbose=True,
      handle_parsing_errors=self._handle_error,
      return_intermediate_steps=True,
    )

    try:
      output = agent_executor(
        {
          "input": self.question,
          "context": self.context,
          "policy": self.retroformer_prompt,
          "long_term_memory": self.format_longterm_memory()
        }
      )
    except ValueError as error:
      print(error)
      output = {
        "input": self.question,
        "context": self.context,
        "policy": self.retroformer_prompt,
        "long_term_memory": self.format_longterm_memory(),
        "output": "Performed a lookup without a search.",
        "intermediate_steps": [],
        }

    self.episode += 1
    print("output", output)
    return self.trail_id, output, self.answer

  def test(self):
    """ 
      Perform run under the current policy.
    """
    agent_executor = AgentExecutor.from_agent_and_tools(
      agent=self.agent, tools=self.tools, verbose=True,
    )
    question = "Which harry potter film series main stars debuted in stage acting first?"
    answer = agent_executor.run(input=question, policy="test policy", long_term_memory=self.format_longterm_memory(), return_intermediate_steps=True)
    print("Answer:", answer)


if __name__ == "__main__":
  actor = Actor(1, 1)
  actor.test()