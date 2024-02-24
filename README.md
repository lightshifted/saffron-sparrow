### How to Use
```bash
curl --location --request POST 'http://localhost:8000/api/generate/records' \
--header 'Content-Type: application/json' \
--header 'Accept: application/json' \
--data-raw '{
    "model":"gpt-3.5-turbo-1106",
    "messages": [
        {
            "role": "user",
            "content": "J30.1"
        }
    ], 
    "cpt_codes": ["76830"]
}'
```