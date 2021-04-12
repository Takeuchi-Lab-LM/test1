from transformers import T5Tokenizer, AutoModelForCausalLM

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

# 推論

strTest = ["グローバリゼーションによって人々の賃金格差は", "ついに完成したこの堤防について、博士は以下のように語っている", "諸行無常のひびきあり","ええ、そのプリン食べちゃったの？"]

for str in strTest:
    input = tokenizer.encode(str, return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=30, num_return_sequences=3)
    print(tokenizer.batch_decode(output))