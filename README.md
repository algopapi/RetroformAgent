# Implementation of the RETRO-FORMER implemenation

- The retroformer model is a vacuna 7B 32k Longchat model using the fastchat API. 

- Currently implementting the Finetune pipeline. 

All credit goes to the original authors of the paper: https://arxiv.org/pdf/2308.02151.pdf

Im just implementing this for the fun of it. (and because i want to see if i can somehow use it in an agent swam)

Stack:
- Actor agents: Langchain
- Agent architecture: MRKL (ReAct Zero Shot)
- Actor model: gpt-4
- Retro model: vacuna 7B
- Environment: HotpotQA


