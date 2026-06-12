# Assembly & Deployment Guide (Phase 6)

## 6.1 Butterfly Canvas Assembly

Use this node sequence in Butterfly:

1. `Upload` (image input)
2. `DocTR + TrOCR` via `POST /process-file`
3. `LayoutLMv3 NER` (auto fallback to regex if fine-tuned weights missing)
4. `BioBERT KB match`
5. `Need Counselor` toggle (boolean)
6. `DSPy counselor` (only when toggle ON and `patient_query` passed)

API output already returns:

- `prescriptions[].medications[]`
- `prescriptions[].counseling` (optional)
- `prescriptions[].action_triggers` (ignore if reminders are out of scope)

## 6.2 Counselor Toggle Behavior

`need_counselor` is now explicit in API input:

- `need_counselor=false` -> no LLM call, only OCR + NER + KB match
- `need_counselor=true` + non-empty `patient_query` -> run DSPy counselor
- `need_counselor=true` + empty `patient_query` -> no counseling output

In Butterfly, bind toggle UI state directly to `need_counselor`.

## 6.3 Edge-case Testing

Recommended test sets:

- heavily distorted scans
- severe spelling errors
- out-of-scope patient questions

Minimum acceptance checks:

- no section headers in extracted drug list
- no KB match for blocked junk tokens
- with `need_counselor=false`, `counseling` is null
- with `need_counselor=true` and question set, `counseling.patient_response` is present
- counseling response stays grounded to `database_facts`

## Local API Run

```bash
pip install -r requirements.txt
uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "page_image/sample_rx.png",
    "ner_backend": "auto",
    "need_counselor": true,
    "patient_query": "Can I take this with food?",
    "counselor_model": "qwen2.5-0.5b-instruct"
  }'
```

Butterfly-style multipart request:

```bash
curl -X POST "http://localhost:8000/process-file" \
  -F "file=@page_image/sample_rx.png" \
  -F "need_counselor=true" \
  -F "patient_query=Can I take this with food?" \
  -F "ner_backend=auto" \
  -F "counselor_model=qwen2.5-0.5b-instruct"
```

## Docker Deployment

```bash
docker compose up --build
```

Health check:

```bash
curl http://localhost:8000/health
```
