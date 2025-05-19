import json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = json.load(open("config/labels.json"))
ID2LABEL = {v: k for k, v in LABELS.items()}

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()


def classify(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
    with torch.no_grad():
        logits = model(**inputs).logits
    return ID2LABEL[int(logits.argmax(-1))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest", action="store_true", help="start FastAPI server")
    args = ap.parse_args()

    if args.rest:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI()


        class Msg(BaseModel):
            text: str


        @app.post("/intent")
        def intent(m: Msg):
            return {"intent": classify(m.text)}


        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        while True:
            txt = input("📝  ")
            print("➡️ ", classify(txt))
