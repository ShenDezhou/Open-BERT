import sys

from transformers import pipeline, BertTokenizer, BertForSequenceClassification

if __name__ == "__main__":
    model_name = sys.argv[1]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True)
    classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)
    response=classifier('很高兴能与你学习', candidate_labels=['正', "负"])
    print(response)