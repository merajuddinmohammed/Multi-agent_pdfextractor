import asyncio
import json
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

from agents.summary_agent import create_agent as create_summary_agent
from agents.action_agent import create_agent as create_action_agent
from agents.risk_agent import create_agent as create_risk_agent
from pdf_parser import pdf_to_chunks

load_dotenv()

app = FastAPI(title="Multi-Agent Document Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.getenv("MODEL_NAME", "arcee-ai/trinity-large-preview:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"raw": text}


async def run_single_agent(agent, message: str) -> dict:
    response = await agent.on_messages(
        [TextMessage(content=message, source="orchestrator")],
        cancellation_token=None,
    )
    return parse_json(response.chat_message.content)


async def run_all_agents(document_chunks: list[str]) -> dict:
    global_context = {"entities": [], "decisions": [], "constraints": []}
    model_client = get_model_client()

    message = json.dumps({
        "document_chunks": document_chunks,
        "global_context": global_context,
    }, indent=2)

    summary_agent = create_summary_agent(model_client)
    action_agent = create_action_agent(model_client)
    risk_agent = create_risk_agent(model_client)

    summary_result, action_result, risk_result = await asyncio.gather(
        run_single_agent(summary_agent, message),
        run_single_agent(action_agent, message),
        run_single_agent(risk_agent, message),
    )

    return {
        "summary": summary_result,
        "actions": action_result,
        "risks": risk_result,
    }


# ---- API Routes ----

@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        chunks = pdf_to_chunks(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    start = time.time()
    results = await run_all_agents(chunks)
    elapsed = round(time.time() - start, 2)

    return {
        "filename": file.filename,
        "chunks_processed": len(chunks),
        "processing_time_seconds": elapsed,
        "results": results,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
