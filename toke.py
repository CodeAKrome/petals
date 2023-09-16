#!env python3
# import torch
# from transformers import AutoTokenizer
# from petals import AutoDistributedModelForCausalLM
# import os
#--
import os
from langchain.llms import Petals
from langchain import PromptTemplate, LLMChain


os.environ["HUGGINGFACE_API_KEY"] = ""

model_name = "meta-llama/Llama-2-70b-chat-hf"

def query(question):
    llm = Petals(model_name="meta-llama/Llama-2-70b-chat-hf")
    #llm = Petals(model_name="bigscience/bloom-petals")
    template = """"Question: {question}
    
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    #print(f"prom: --{prompt}--\n")
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain.run(question))
    

# model_name = "enoch/llama-65b-hf"
# You could also use "meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf", or
# "bigscience/bloom" - basically, any Hugging Face Hub repo with a supported model architecture

def tokes():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
    text = "MOSCOW, June 24. /TASS/. Russian forces are carrying out required operational and combat measures in the southwestern Voronezh Region as part of a counter-terror operation, Regional Governor Alexander Gusev said on Saturday. \"The Russian Armed Forces are carrying out required operational and combat measures on the territory of the Voronezh Region as part of a counter-terror operation. I will keep informing you about the latest developments,\" the governor said on his Telegram channel. A counter-terror regime was introduced in Moscow, the Moscow and Voronezh Regions earlier on Saturday. The Telegram channel of Wagner private military company founder Yevgeny Prigozhin earlier posted several audio records with accusations against the country\u2019s military leaders. In the wake of this, the Federal Security Service (FSB) of Russia has opened a criminal case into a call for an armed mutiny. The FSB urged Wagner fighters not to obey Prigozhin\u2019s orders and take measures for his detention."
    tokens = tokenizer.encode(text)
    words = text.split()
    print(f"wurd: {len(words)} toke: {len(tokens)}")

# ---

question = "What happened on June 24?"
query(question)
