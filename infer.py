
import json
import argparse
from itertools import chain

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np

def tokenize(example, tokenizer, max_length):
    text = []
    token_map = []
    
    idx = 0
    
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        
        text.append(t)
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
            
        idx += 1
            
        
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)
    
        
    return {
        **tokenized,
        "token_map": token_map,
    }

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int)
    
    args = parser.parse_args()
    
    data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/test.json"))
    
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = ds.map(
        tokenize, 
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length}, 
        num_proc=2,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=4, 
        report_to="none",
    )
    
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )
    
    
    predictions = trainer.predict(ds).predictions

    ds.to_parquet("test_ds.pq")
    
    np.save("preds.npy", predictions)
    
    
if __name__ == "__main__":
    main()
