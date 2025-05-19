# multiintent-lite

A tiny multilingual intent classifier (Greeting / Farewell / Toxic / Chitchat) that
runs on CPU in < 100 MB RAM and handles ~150 req/s on a 4-core VM.

## Quick start
```bash
git clone …
cd multiintent-lite
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/run_train.sh      # fine-tunes model
bash scripts/run_infer.sh      # launches REST API
````

---

### 📌 How to integrate with Dify

1. **Host predict.py** (REST mode) somewhere reachable (same server is fine).  
2. In your Dify Workflow add an **HTTP Request** node:
   - Method : POST  
   - URL : `http://your-server:8000/intent`  
   - Body : `{ "text": "{{sys.query}}" }`  
3. Follow that with a **Conditional Branch** node:
   - IF `response.intent == "Greeting"` → Greeting path  
   - IF … etc.  
