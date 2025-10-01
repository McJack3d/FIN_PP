"""FastAPI app for Alstom predictor. Launch with: uvicorn ALST_PP.webapi.main:app --reload"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from ALST_PP.interfaces.cli.ALST_Ticker_Predictor_Final import run

templates = Jinja2Templates(directory=str(Path(__file__).parent / "template"))

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ... rest of the file unchanged ...
