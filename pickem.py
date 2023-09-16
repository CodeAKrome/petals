#!env python
import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import os
import sys

os.environ["HUGGINGFACE_API_KEY"] = ""

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "meta-llama/Llama-2-70b-chat-hf"

print(f"MODEL: {model_name}")

# model_name = "enoch/llama-65b-hf"
# You could also use "meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf", or
# "bigscience/bloom" - basically, any Hugging Face Hub repo with a supported model architecture

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
model = model.cuda()

fake_token = tokenizer("^")["input_ids"][0]  # Workaround to make tokenizer.decode() keep leading spaces

cnt = 0

with model.inference_session(max_length=4096) as sess:
    while True:
        prompt = input('Human: ')
        if prompt == "":
            break
        prefix = f"Human: {prompt}\nFriendly AI:"
        prefix = tokenizer(prefix, return_tensors="pt")["input_ids"].cuda()
        print("Friendly AI:", end="", flush=True)

        while True:
            outputs = model.generate(prefix, max_new_tokens=1, session=sess,
                                     do_sample=True, temperature=0.9, top_p=0.6)
            outputs = tokenizer.decode([fake_token, outputs[0, -1].item()])[1:]

            # Now, let's print one new token at a time
            print(outputs, end="", flush=True)

            if "\n" in outputs:
                cnt += 1
                if cnt > 7:
                    break
            prefix = None  # Prefix is passed only for the 1st token of the bot's response
