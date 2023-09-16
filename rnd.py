#!env python
import torch
from transformers import CodeLlamaTokenizer
from petals import AutoDistributedModelForCausalLM
import os

os.environ["HUGGINGFACE_API_KEY"] = ""

model_name = "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = CodeLlamaTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
# model = model.cuda()

# ckpt_dir = '/home/kyle/srcLocal/codellama/CodeLlama-7b-Instruct/'
# tokenizer_path = '/home/kyle/srcLocal/codellama/CodeLlama-7b-Instruct/tokenizer.model'
# max_seq_len = 512
# max_batch_size = 4

instructions = [
    [
        {
            "role": "user",
            "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
        }
    ],
    [
        {
            "role": "user",
            "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
        }
    ],
    [
        {
            "role": "system",
            "content": "Provide answers in JavaScript",
        },
        {
            "role": "user",
            "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
        },
    ],
]

instant = """
[SYS]
[INST]
role: system
content: Provide answers in Python
[/INST]
[INST]
role: user
content: In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?
[/INST]
[/SYS]
"""

# inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
inputs = tokenizer(instant, return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))
