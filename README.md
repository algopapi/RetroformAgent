# RETROFORMER: RETROSPECTIVE LARGE LANGUAGE
AGENTS WITH POLICY GRADIENT OPTIMIZATION - Implementation

- The retroformer model is a longchat-7b-32k model using the fastchat API. 

- Currently implementting the Finetune pipeline. 

All credit goes to the original authors of the paper: https://arxiv.org/pdf/2308.02151.pdf

Im just implementing this for the fun of it. (and because i want to see if i can somehow use it in an agent swam)

Stack:
- Actor agents: Langchain
- Agent architecture: MRKL (ReAct Zero Shot)
- Actor model: gpt-4
- Retro model: longchat-7b-32k
- Environment: HotpotQA


To start the locally hosted server:

python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" --model-path lmsys/longchat-7b-32k
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000