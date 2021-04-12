from transformers import T5Tokenizer, AutoModelForCausalLM

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

# 推論
input = tokenizer.encode("グローバリゼーションによって人々の賃金格差は", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=1)
print(tokenizer.batch_decode(output))