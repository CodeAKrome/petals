from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import sys

# Choose any model available at https://health.petals.dev
model_name = "petals-team/StableBeluga2"
model_name = "tiiuae/falcon-180B-chat"
#model_name = sys.argv[1]

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...
