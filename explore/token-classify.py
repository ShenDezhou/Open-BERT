from transformers import pipeline, BertTokenizer, BertForTokenClassification

tokenizer = BertTokenizer.from_pretrained('chinese_L-2_H-128_A-2')
model = BertForTokenClassification.from_pretrained('chinese_L-2_H-128_A-2', return_dict=True)
classifier = pipeline('token-classification', model=model, tokenizer=tokenizer)
response=classifier('We are very happy to show you the 🤗 Transformers library.')
print(response)