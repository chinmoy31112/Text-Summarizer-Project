from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from textSummarizer.pipeline.prediction import PredictionPipeline
import uvicorn

app = FastAPI(title="Text Summarizer API", description="Summarize text using PEGASUS")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    """Accept text input and return the generated summary."""
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": summary, "original_text": text},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e), "original_text": text},
        )


@app.get("/train")
async def train():
    """Trigger the full training pipeline."""
    import os
    os.system("python main.py")
    return RedirectResponse(url="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
