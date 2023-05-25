from transformers import pipeline, BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('chinese_L-2_H-128_A-2')
model = BertModel.from_pretrained('chinese_L-2_H-128_A-2', return_dict=True)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
response=classifier('We are very happy to show you the 🤗 Transformers library.')
print(response)