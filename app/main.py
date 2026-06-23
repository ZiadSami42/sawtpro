import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types  # Imported for structural generation configs
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the new Google GenAI Client
# The Client will automatically pick up GEMINI_API_KEY from the environment,
# but we maintain your explicit validation check below.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    if not request.text or len(request.text) < 15:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    prompt = f"""
        أنت خبير في اللغة العربية والتعليق الصوتي.
        إليك هذا النص الذي سيقرأه معلق صوتي:
        "{request.text}"
        
        المطلوب منك:
        1. تشكيل النص لغوياً بالكامل وبدقة شديدة (الإعراب الصحيح) ليقرأه المعلق بدون أي أخطاء نحوية.
        2. تحليل النص واقتراح النبرة أو اللون الصوتي الأنسب لقراءته، اختر فقط واحدة من: (وثائقي، إعلاني، تمثيلي).
        3. كتابة سبب قصير (جملة واحدة) لاختيارك لهذه النبرة.
        
        Return the response as a JSON object with keys: vocalizedText, suggestedTone, reasoning.
    """

    try:
        # Use the async client (.aio) to avoid blocking the FastAPI event loop
        response = await client.aio.models.generate_content(
            model='gemini-3.1-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))