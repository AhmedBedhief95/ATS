import io
import json
import os
import uvicorn
import pdfplumber
import auth
import requests
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import types
from local_analyzer_wrapper import get_local_analyzer
from config import config
from collections import Counter

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Groq
if config.GROQ_API_KEY:
    client = Groq(api_key=config.GROQ_API_KEY)
else:
    client = None

# Initialize Gemini
if config.GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
else:
    gemini_client = None

def get_error_result(error_msg):
    return {
        "name": "Analysis Error",
        "job": "Processing Failed",
        "experience": 0,
        "score": 0,
        "summary": f"Error: {error_msg}",
        "key_strengths": [],
        "identified_gaps": ["Check server logs"],
        "email": "Not found",
        "phone": "Not found",
        "top_skills": [],
        "all_skills_found": [],
        "suggested_job": "Not available",
        "suggested_salary": 0,
        "residence": "Not specified"
    }

# --- Page Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/hr_dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/admin_dashboard", response_class=HTMLResponse)
async def admin_dashboard():
    with open("admin.html", "r") as f:
        return f.read()

@app.get("/manager_dashboard", response_class=HTMLResponse)
async def manager_dashboard():
    with open("manager_dashboard.html", "r") as f:
        return f.read()

# --- Admin API ---
@app.get("/api/admin/users")
async def api_get_admin_users():
    return auth.get_all_users()

@app.post("/api/admin/create_user")
async def api_create_user(data: dict):
    success = auth.create_new_user(data)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to create user")

@app.post("/api/admin/update_user/{user_id}")
async def api_update_user(user_id: int, data: dict):
    success = auth.update_user_details(user_id, data)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to update user")

# --- Candidate APIs (History & Recommendations) ---
@app.get("/api/candidates")
async def api_get_db_candidates():
    return auth.get_all_candidates()

@app.get("/api/recommendations")
async def api_get_recommendations():
    return auth.get_all_recommendations()

@app.post("/api/save_candidates")
async def save_all_candidates(candidates: list[dict]):
    """Save all candidates to candidates table (history)."""
    try:
        count = auth.save_candidates_to_db(candidates)
        return {"status": "success", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save_recommendations")
async def save_recommendations(candidates: list[dict]):
    """Save selected candidates to recommendation table."""
    try:
        result = auth.save_candidates_to_recommendation(candidates)
        return {"status": "success", "count": result["count"], "rec_nums": result["rec_nums"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Recommendation Details & Actions ---
@app.get("/api/recommendations/{rec_num}/details")
async def get_recommendation_details(rec_num: int):
    details = auth.get_recommendation_details(rec_num)
    if not details:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return details

@app.get("/api/recommendations/{rec_num}/contact")
async def get_contact_info(rec_num: int):
    contact = auth.get_contact_info(rec_num)
    if not contact:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return contact

@app.post("/api/recommendations/{rec_num}/review")
async def update_review_status(rec_num: int):
    success = auth.update_review_status(rec_num)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"status": "success"}

# --- CPI Email Notification ---
class NotificationRequest(BaseModel):
    action: str
    message: str

@app.post("/api/recommendations/{rec_num}/send-notification")
async def send_candidate_notification(rec_num: int, req: NotificationRequest):
    details = auth.get_recommendation_details(rec_num)
    if not details:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    contact = auth.get_contact_info(rec_num)
    if not contact or not contact.get('email') or contact['email'] == 'Not provided':
        raise HTTPException(status_code=400, detail="No valid email address")
    email = contact['email']
    first_name = details.get('name', 'Candidate').split()[0]
    soap_template = f'''<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/">
   <soapenv:Header/>
   <soapenv:Body>
      <m:SendNotification xmlns:m="http://example.com/mail">
         <m:Recipient>
            <m:EmailAddress>{email}</m:EmailAddress>
            <m:FirstName>{first_name}</m:FirstName>
         </m:Recipient>
         <m:Content>{req.message}</m:Content>
      </m:SendNotification>
   </soapenv:Body>
</soapenv:Envelope>'''
    cpi_url = config.CPI_BASE_URL + "/cxf/saop_sender"
    auth_basic = HTTPBasicAuth(config.CPI_CLIENT_ID, config.CPI_CLIENT_SECRET)
    headers = {'Content-Type': 'text/xml; charset=utf-8'}
    try:
        response = requests.post(cpi_url, data=soap_template.encode('utf-8'), headers=headers, auth=auth_basic, timeout=30)
        response.raise_for_status()
        return {"success": True, "message": "Notification sent successfully"}
    except Exception as e:
        print(f"SOAP send error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send: {str(e)}")

# --- Manager Dashboard Aggregated Data ---
@app.get("/api/manager/dashboard")
async def get_manager_dashboard_data():
    candidates = auth.get_all_candidates()
    total_candidates = len(candidates)
    if candidates:
        top_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]
    else:
        top_candidates = []
    monthly_revenue = [12500, 14200, 16800, 15200, 18900, 21000, 23500, 24800, 26700, 28200, 30100, 32500]
    cat_counter = Counter([c.get('job', 'Unknown') for c in candidates])
    category_data = [{"name": k, "value": v} for k, v in cat_counter.most_common(6)]
    if not category_data:
        category_data = [{"name": "No Data", "value": 1}]
    return {
        "totalIncome": 346042,
        "spendingIncome": 4329,
        "historyTotal": total_candidates,
        "billable": 65,
        "nonBillable": 35,
        "topCandidates": top_candidates,
        "monthlyRevenue": monthly_revenue,
        "categoryData": category_data
    }

# --- Analysis Endpoint (updated with residence) ---
@app.post("/analyze")
async def analyze(jd: str = Form(...), model: str = Form("groq"), files: list[UploadFile] = File(...)):
    results = []
    local_analyzer = None

    for file in files:
        content = await file.read()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join([p.extract_text() or "" for p in pdf.pages])

        if not text.strip():
            results.append({
                "name": "Empty PDF", "job": "Extraction Failed", "experience": 0, "score": 0,
                "summary": "Could not extract text from PDF.", "key_strengths": [], "identified_gaps": [],
                "email": "Not found", "phone": "Not found", "top_skills": [], "all_skills_found": [],
                "job_name": jd, "model_used": model,
                "suggested_job": "Not available", "suggested_salary": 0,
                "residence": "Not specified"
            })
            continue

        try:
            if model == "groq" and client:
                prompt = (
                    f"Analyze this resume based on the Job Description: {jd}.\n"
                    f"Resume Text: {text}.\n"
                    "Return ONLY a JSON object with these keys: "
                    "name, job, email, phone, top_skills (list), all_skills_found (list), "
                    "experience (numeric years), score (0-100), summary, key_strengths (list), identified_gaps (list), "
                    "suggested_job (string), suggested_salary (integer), residence (string)."
                )
                res = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"}
                )
                analysis_data = json.loads(res.choices[0].message.content)
                analysis_data.setdefault("email", "Not found")
                analysis_data.setdefault("phone", "Not found")
                analysis_data.setdefault("top_skills", [])
                analysis_data.setdefault("all_skills_found", [])
                analysis_data.setdefault("suggested_job", "Not specified")
                analysis_data.setdefault("suggested_salary", 0)
                analysis_data.setdefault("residence", "Not specified")

            elif model == "gemini" and gemini_client:
                prompt = (
                    f"Analyze this resume based on the Job Description: {jd}.\n"
                    f"Resume Text: {text}.\n"
                    "Return ONLY a JSON object with these keys: "
                    "name, job, email, phone, top_skills (list), all_skills_found (list), "
                    "experience (numeric years), score (0-100), summary, key_strengths (list), identified_gaps (list), "
                    "suggested_job (string), suggested_salary (integer), residence (string)."
                )
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    response_text = response.text
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0]
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0]
                    analysis_data = json.loads(response_text)
                except Exception as e:
                    print(f"Gemini error: {e}")
                    analysis_data = get_error_result(str(e))
                analysis_data.setdefault("email", "Not found")
                analysis_data.setdefault("phone", "Not found")
                analysis_data.setdefault("top_skills", [])
                analysis_data.setdefault("all_skills_found", [])
                analysis_data.setdefault("suggested_job", "Not specified")
                analysis_data.setdefault("suggested_salary", 0)
                analysis_data.setdefault("residence", "Not specified")

            elif model == "local":
                if local_analyzer is None:
                    local_analyzer = get_local_analyzer()
                analysis_data = local_analyzer.analyze_resume(text, jd)
                analysis_data.setdefault("email", "Not found")
                analysis_data.setdefault("phone", "Not found")
                analysis_data.setdefault("top_skills", [])
                analysis_data.setdefault("all_skills_found", [])
                analysis_data.setdefault("key_strengths", [])
                analysis_data.setdefault("identified_gaps", [])
                analysis_data.setdefault("suggested_job", "Not specified")
                analysis_data.setdefault("suggested_salary", 0)
                analysis_data.setdefault("residence", "Not specified")

            else:
                if client:
                    prompt = (
                        f"Analyze this resume based on the Job Description: {jd}.\n"
                        f"Resume Text: {text}.\n"
                        "Return ONLY a JSON object with these keys: "
                        "name, job, email, phone, top_skills (list), all_skills_found (list), "
                        "experience (numeric years), score (0-100), summary, key_strengths (list), identified_gaps (list), "
                        "suggested_job (string), suggested_salary (integer), residence (string)."
                    )
                    res = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                        response_format={"type": "json_object"}
                    )
                    analysis_data = json.loads(res.choices[0].message.content)
                else:
                    raise Exception("No AI model available")
                analysis_data.setdefault("email", "Not found")
                analysis_data.setdefault("phone", "Not found")
                analysis_data.setdefault("top_skills", [])
                analysis_data.setdefault("all_skills_found", [])
                analysis_data.setdefault("suggested_job", "Not specified")
                analysis_data.setdefault("suggested_salary", 0)
                analysis_data.setdefault("residence", "Not specified")

            analysis_data["job_name"] = jd
            analysis_data["model_used"] = model
            results.append(analysis_data)

        except Exception as e:
            print(f"Error on {file.filename}: {e}")
            results.append({
                "name": f"Error: {file.filename}", "job": "Analysis Failed", "experience": 0, "score": 0,
                "summary": str(e), "key_strengths": [], "identified_gaps": [],
                "email": "Not found", "phone": "Not found", "top_skills": [], "all_skills_found": [],
                "job_name": jd, "model_used": model,
                "suggested_job": "Error", "suggested_salary": 0,
                "residence": "Not specified"
            })

    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return results

# --- Model Info ---
@app.get("/api/model-info")
async def get_model_info():
    local_analyzer = get_local_analyzer()
    local_info = local_analyzer.get_model_info()
    return {
        "models": {
            "groq": {"available": bool(config.GROQ_API_KEY), "name": "Groq LLM"},
            "gemini": {"available": bool(config.GEMINI_API_KEY), "name": "Google Gemini"},
            "local": {"available": local_info["type"] == "trained_random_forest", "name": "Local Trained AI", "details": local_info}
        }
    }

# --- Auth ---
@app.post("/login_check")
async def login(data: dict):
    user = auth.check_credentials(data.get("username"), data.get("password"), data.get("role"))
    if user:
        if not user[2]:
            secret, qr = auth.generate_new_qr(user[0])
            return {"status": "enroll_2fa", "username": user[0], "role": user[1], "secret": secret, "qr_code": qr}
        return {"status": "verify_2fa", "username": user[0], "role": user[1], "secret": user[2]}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/verify_2fa_step")
async def verify(data: dict):
    if auth.verify_2fa(data.get("secret"), data.get("code")):
        if data.get("is_enrollment"):
            auth.update_user_secret(data.get("username"), data.get("secret"))
        return {"status": "verified"}
    raise HTTPException(status_code=400, detail="Invalid 2FA code")

@app.on_event("startup")
async def startup_event():
    print("\n🚀 ATS+ Server Starting...")
    print(f"  Groq: {'✅' if config.GROQ_API_KEY else '❌'}")
    print(f"  Gemini: {'✅' if config.GEMINI_API_KEY else '❌'}")
    local_analyzer = get_local_analyzer()
    print(f"  Local AI: {'✅ Trained' if local_analyzer.use_trained_model else '⚠️ Pattern only'}")
    print(f"📍 http://{config.APP_HOST}:{config.APP_PORT}\n")

if __name__ == "__main__":
    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT)