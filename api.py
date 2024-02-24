import os
from typing import List, Union

from datasets import load_dataset
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import generate

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

codebank = load_dataset("jilp00/enriched-icdcm-data", token=HF_TOKEN).rename_column("codes", "code")
seed_data = load_dataset("jilp00/enhanced-medical-records", token=HF_TOKEN)

# HOME_URL = "https://aapc-datagen.centralus.cloudapp.azure.com/"
HOME_URL = "http://localhost:3000/"


app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        HOME_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo-1106"
    messages: List[Message]
    cpt_codes: Union[List[str], str]


@app.post("/api/generate/records")
async def chat_completion(request: ChatRequest):
    codes = request.messages[0].content
    model = request.model
    cpt_codes = request.cpt_codes

    record = generate(user_codes=codes, procedure_code=cpt_codes, codebank=codebank["train"], seed_data=seed_data["train"], model=model)

    return record


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
