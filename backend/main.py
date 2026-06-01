import sys
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Fix Playwright asyncio issue on Windows
# Use SelectorEventLoop for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="ResearchFlow API",
    description="ResearchFlow Backend API for document management and RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api import documents, chat, search, memory, ocr

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["ocr"])

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
