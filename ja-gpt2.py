from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

input = tokenizer.encode("グローバリゼーションによって人々の賃金格差は", return_tensors="pt")

#do_sample=False(default),use greedy decoding insteadof sampling
#no_repeat_ngram_size, all ngrams of that size can only occur once
output = model.generate(input,do_sample=False,max_length=100,num_beams=5,no_repeat_ngram_size=2)

#skip_special_tokensremove special tokens in the decoding
print(tokenizer.batch_decode(output,skip_special_tokens=True))

'''グローバリゼーションによって人々の賃金格差は拡大の一途をたどっています。この格差を是正するためには、
 労働者の賃金を引き上げることが必要です。しかし、賃金の引き上げは簡単ではありません。なぜなら、多くの労働者
 は賃金が上がらなければ生活が成り立たなくなるのですから。 では、どうすれば賃金を上げることができるのでしょう
 か? 今回は「賃金引き上げの方法」について考えてみたいと思います。 「賃金」とは、労働'''