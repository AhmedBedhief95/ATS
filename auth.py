import psycopg2
import pyotp
import qrcode
import io
import base64
import json
import random
import traceback
from config import config

DB_CONFIG = config.DB_CONFIG

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

# ---------------------- USER MANAGEMENT ----------------------
def check_credentials(username, password, role):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT usr_name, role, otp_secret FROM users WHERE usr_name = %s AND usr_password = %s AND role = %s", (username, password, role))
                return cur.fetchone()
    except Exception as e:
        print(f"DB Auth Error: {e}")
        return None

def create_new_user(data):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (usr_name, usr_last_name, role, usr_password, lock)
                    VALUES (%s, %s, %s, %s, %s)
                """, (data.get("usr_name"), data.get("usr_last_name"), int(data.get("role")), data.get("usr_password"), False))
                conn.commit()
                return True
    except Exception as e:
        print(f"Create User Error: {e}")
        return False

def update_user_details(user_id, data):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                otp_val = None if data.get("clear_2fa") else data.get("otp_secret")
                cur.execute("""
                    UPDATE users SET usr_name = %s, usr_last_name = %s, role = %s, otp_secret = %s, lock = %s WHERE id = %s
                """, (data.get("usr_name"), data.get("usr_last_name"), int(data.get("role")), otp_val, bool(data.get("lock")), user_id))
                conn.commit()
                return True
    except Exception as e:
        print(f"Update User Error: {e}")
        return False

def get_all_users():
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, usr_name, usr_last_name, role, otp_secret, lock FROM users ORDER BY id ASC")
                rows = cur.fetchall()
                role_map = {1: "Administrator", 2: "HR Professional", 3: "Head Manager"}
                return [{"id": r[0], "usr_name": r[1], "usr_last_name": r[2], "role": r[3],
                         "role_name": role_map.get(r[3], "Unknown"), "has_2fa": bool(r[4]), "lock": r[5]} for r in rows]
    except Exception as e:
        print(f"Get users error: {e}")
        return []

# ---------------------- CANDIDATES TABLE (history) ----------------------
def get_all_candidates():
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT first_name, last_name, current_job, experience, score, job_name,
                           sug_job, salary, residence
                    FROM candidates ORDER BY score DESC
                """)
                rows = cur.fetchall()
                result = []
                for r in rows:
                    result.append({
                        "name": f"{r[0]} {r[1]}" if r[0] and r[1] else r[0] or r[1] or "Unknown",
                        "job": r[2] if r[2] else "Not Specified",
                        "exp": int(r[3]) if r[3] else 0,
                        "score": int(r[4]) if r[4] else 0,
                        "target_job": r[5] if r[5] else "",
                        "sug_job": r[6] if r[6] else "Not specified",
                        "salary": int(r[7]) if r[7] else 0,
                        "residence": r[8] if r[8] else "Not specified"
                    })
                return result
    except Exception as e:
        print(f"Get candidates error: {e}")
        return []

def save_candidates_to_db(candidates):
    """Save all candidates to the candidates table (history) without generating rec_num."""
    count = 0
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                for candidate in candidates:
                    # Name split
                    full_name = candidate.get('name', 'Unknown')
                    name_parts = full_name.split() if full_name else ['Unknown', 'Unknown']
                    first_name = name_parts[0] if len(name_parts) > 0 else 'Unknown'
                    last_name = name_parts[-1] if len(name_parts) > 1 else ''

                    # Experience & score
                    experience = candidate.get('experience', 0)
                    try:
                        experience = int(float(experience))
                    except:
                        experience = 0
                    score = candidate.get('score', 0)
                    try:
                        score = int(float(score))
                    except:
                        score = 0

                    current_job = candidate.get('job', candidate.get('current_job', 'Not Specified'))
                    job_name = candidate.get('job_name', '')
                    sug_job = candidate.get('suggested_job', 'Not specified')
                    salary = candidate.get('suggested_salary', 0)
                    if salary is None:
                        salary = 0
                    salary = int(salary)
                    residence = candidate.get('residence', 'Not specified')

                    cur.execute("""
                        INSERT INTO candidates (
                            first_name, last_name, current_job, experience, score, job_name,
                            sug_job, salary, residence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (first_name, last_name, current_job, experience, score, job_name,
                          sug_job, salary, residence))
                    count += 1
                conn.commit()
                return count
    except Exception as e:
        print(f"Save to candidates error: {e}")
        traceback.print_exc()
        raise e

# ---------------------- RECOMMENDATION TABLE (manager review) ----------------------
def get_all_recommendations():
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT first_name, last_name, current_job, experience, score, job_name,
                           rec_num, review, accept, email, sug_job, salary, residence
                    FROM recommendation ORDER BY score DESC
                """)
                rows = cur.fetchall()
                result = []
                for r in rows:
                    result.append({
                        "name": f"{r[0]} {r[1]}" if r[0] and r[1] else r[0] or r[1] or "Unknown",
                        "job": r[2] if r[2] else "Not Specified",
                        "exp": int(r[3]) if r[3] else 0,
                        "score": int(r[4]) if r[4] else 0,
                        "target_job": r[5] if r[5] else "",
                        "rec_num": r[6],
                        "review": "Reviewed" if r[7] else "Pending",
                        "accept": "Accepted" if r[8] is True else ("Refused" if r[8] is False else "—"),
                        "email": r[9] if r[9] else "Not provided",
                        "sug_job": r[10] if r[10] else "Not specified",
                        "salary": int(r[11]) if r[11] else 0,
                        "residence": r[12] if r[12] else "Not specified"
                    })
                return result
    except Exception as e:
        print(f"Get recommendations error: {e}")
        traceback.print_exc()
        return []

def get_recommendation_details(rec_num):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT strengths, gaps, skills, first_name, last_name, current_job, score, sug_job, salary, residence
                    FROM recommendation WHERE rec_num = %s
                """, (rec_num,))
                row = cur.fetchone()
                if row:
                    strengths = json.loads(row[0]) if row[0] else []
                    gaps = json.loads(row[1]) if row[1] else []
                    skills = json.loads(row[2]) if row[2] else []
                    return {
                        "name": f"{row[3]} {row[4]}".strip(),
                        "job": row[5] if row[5] else "Not specified",
                        "score": row[6] if row[6] else 0,
                        "strengths": strengths,
                        "gaps": gaps,
                        "skills": skills,
                        "sug_job": row[7] if row[7] else "Not specified",
                        "salary": int(row[8]) if row[8] else 0,
                        "residence": row[9] if row[9] else "Not specified"
                    }
                return None
    except Exception as e:
        print(f"Get details error: {e}")
        return None

def get_contact_info(rec_num):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT email, phone FROM recommendation WHERE rec_num = %s", (rec_num,))
                row = cur.fetchone()
                if row:
                    return {"email": row[0] or "Not provided", "phone": row[1] or "Not provided"}
                return None
    except Exception as e:
        print(f"Get contact error: {e}")
        return None

def update_review_status(rec_num):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE recommendation SET review = TRUE WHERE rec_num = %s", (rec_num,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print(f"Update review error: {e}")
        return False

def save_candidates_to_recommendation(candidates):
    """Save selected candidates to the recommendation table with unique rec_num."""
    saved_count = 0
    assigned_rec_nums = []
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                for candidate in candidates:
                    # Generate unique 6-digit rec_num
                    while True:
                        rec_num = random.randint(100000, 999999)
                        cur.execute("SELECT id FROM recommendation WHERE rec_num = %s", (rec_num,))
                        if not cur.fetchone():
                            break
                    # Name split
                    full_name = candidate.get('name', 'Unknown')
                    name_parts = full_name.split() if full_name else ['Unknown', 'Unknown']
                    first_name = name_parts[0] if len(name_parts) > 0 else 'Unknown'
                    last_name = name_parts[-1] if len(name_parts) > 1 else ''
                    # Experience & score
                    experience = candidate.get('experience', 0)
                    try:
                        experience = int(float(experience))
                    except:
                        experience = 0
                    score = candidate.get('score', 0)
                    try:
                        score = int(float(score))
                    except:
                        score = 0
                    current_job = candidate.get('job', candidate.get('current_job', 'Not Specified'))
                    job_name = candidate.get('job_name', '')
                    email = candidate.get('email', 'Not provided')
                    phone = candidate.get('phone', 'Not provided')
                    # JSON fields
                    strengths = candidate.get('key_strengths', candidate.get('strengths', [])) or []
                    gaps = candidate.get('identified_gaps', candidate.get('gaps', [])) or []
                    skills = candidate.get('top_skills', candidate.get('skills', [])) or []
                    strengths_json = json.dumps(strengths if isinstance(strengths, list) else [])
                    gaps_json = json.dumps(gaps if isinstance(gaps, list) else [])
                    skills_json = json.dumps(skills if isinstance(skills, list) else [])
                    # New fields
                    sug_job = candidate.get('suggested_job', 'Not specified')
                    salary = candidate.get('suggested_salary', 0)
                    if salary is None:
                        salary = 0
                    salary = int(salary)
                    residence = candidate.get('residence', 'Not specified')

                    cur.execute("""
                        INSERT INTO recommendation (
                            rec_num, first_name, last_name, current_job, phone, email,
                            experience, score, job_name, strengths, gaps, skills,
                            sug_job, salary, residence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (rec_num, first_name, last_name, current_job, phone, email,
                          experience, score, job_name, strengths_json, gaps_json, skills_json,
                          sug_job, salary, residence))
                    saved_count += 1
                    assigned_rec_nums.append(rec_num)
                conn.commit()
                return {"count": saved_count, "rec_nums": assigned_rec_nums}
    except Exception as e:
        print(f"Save to recommendation error: {e}")
        traceback.print_exc()
        raise e

# ---------------------- 2FA ----------------------
def update_user_secret(username, secret):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET otp_secret = %s WHERE usr_name = %s", (secret, username))
                conn.commit()
    except Exception as e:
        print(f"Update Secret Error: {e}")

def verify_2fa(secret, code):
    return pyotp.TOTP(secret).verify(code, valid_window=1)

def generate_new_qr(username):
    secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="ATS+")
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return secret, f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"