import sys

from transformers import pipeline, BertTokenizer, BertForMaskedLM

if __name__ == "__main__":
    model_name = sys.argv[1]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name, return_dict=True)

    mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    text = '奶茶里面有[MASK]子'
    response = mask(text)
    print(response)
