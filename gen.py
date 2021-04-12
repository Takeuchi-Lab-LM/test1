import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

seq = ("グローバリゼーションによって人々の賃金格差は")

inputs = tokenizer.encode(seq, return_tensors='pt')
outputs = model.generate(inputs, do_sample=True, max_length=100, num_return_sequences=5)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)