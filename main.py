import os
import time
import json
import uuid
import tempfile
import hashlib
import requests
import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="RateEngine API (PDF -> Gemini -> Cloudflare D1)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict this to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_DATABASE_ID = os.getenv("CF_DATABASE_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")

# ==========================================
# HELPER: CLOUDFLARE D1 DATABASE EXECUTION
# ==========================================
def execute_d1_query(sql: str, params: list = None):
    """Executes a query against your Cloudflare D1 Database via REST API"""
    if not all([CF_ACCOUNT_ID, CF_DATABASE_ID, CF_API_TOKEN]):
        print("Warning: Cloudflare credentials missing. Skipping actual DB execution.")
        return []

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{CF_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {"sql": sql}
    if params:
        payload["params"] = params

    response = requests.post(url, headers=headers, json=payload)
    
    if not response.ok:
        raise Exception(f"D1 API Error: {response.text}")
        
    result = response.json()
    if not result.get("success"):
         raise Exception(f"D1 Query Failed: {result.get('errors')}")
         
    return result["result"][0].get("results", [])

# ==========================================
# HELPER: GEMINI AI SCHEMA MAPPER
# ==========================================
def get_ai_schema_mapping(sample_rows):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # Restored to the robust prompt that worked for you originally
    system_prompt = """You are a highly adaptable data schema mapper for a general product catalog and pricing application. 
    I will provide a JSON array containing the first 8 rows of an extracted PDF table representing a price list for ANY type of product.
    First, identify which row actually contains the column headers (usually index 0, 1, or 2).
    Then, identify which column index (0-based) corresponds to our core database fields.
    
    CRITICAL: For the "attribute_indices" array, you MUST include ALL remaining column indices that contain product specifications, features, or variants. Do not leave this array empty if there are extra descriptive columns!
    
    Return ONLY a valid JSON object matching this schema exactly:
    {
      "header_row_index": integer,
      "model_name_index": integer, 
      "mrp_index": integer,
      "list_price_ex_gst_index": integer, 
      "list_price_inc_gst_index": integer,
      "attribute_indices": [integer, integer, ...]
    }
    Use -1 if a core field is completely missing."""
    
    payload = {
        "contents": [{"parts": [{"text": f"Sample Context Rows:\n{json.dumps(sample_rows, indent=2)}"}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    response = requests.post(url, json=payload)
    if not response.ok:
        raise Exception(f"AI Provider Error: {response.text}")
        
    return json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])

def clean_price(val):
    if not val: return 0.0
    cleaned = ''.join(c for c in str(val) if c.isdigit() or c == '.')
    try: return float(cleaned) if cleaned else 0.0
    except ValueError: return 0.0

def get_deterministic_id(brand: str, model: str, attrs: dict = None) -> str:
    """Generates a consistent ID including attributes to differentiate variants (e.g., colors, sizes)"""
    unique_str = f"{str(brand).strip().lower()}|{str(model).strip().lower()}"
    
    if attrs:
        # Sort the dictionary items by key to ensure the hash is always identical for the exact same attributes
        sorted_attrs = sorted(attrs.items())
        attr_str = "|".join([f"{str(k).strip().lower()}:{str(v).strip().lower()}" for k, v in sorted_attrs])
        unique_str += f"|{attr_str}"
        
    return hashlib.md5(unique_str.encode()).hexdigest()

# ==========================================
# ENDPOINT: UPLOAD & PROCESS PDF (Heavy lifting only)
# ==========================================
@app.post("/api/upload")
async def upload_pdf(brandName: str = Form(...), file: UploadFile = File(...)):
    """
    1. Receives PDF from Frontend.
    2. Extracts 2D array using pdfplumber.
    3. Maps columns using Gemini API.
    4. Cleans & formats data.
    5. Batch pushes to Cloudflare D1.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file.")
        
    temp_pdf_path = None
    try:
        # Save temp file for pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
            
        all_rows = []
        
        # 1. Extract Grid from PDF
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    cleaned_table = [
                        [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                        for row in table if any(cell for cell in row)
                    ]
                    all_rows.extend(cleaned_table)
                    
        if len(all_rows) < 2:
            raise HTTPException(status_code=400, detail="No readable tabular data found in PDF.")
            
        # 2. Get AI Schema Mapping (Use first 8 rows to catch hidden headers)
        sample_rows = all_rows[:8]
        mapping = get_ai_schema_mapping(sample_rows)
        
        header_idx = mapping.get("header_row_index", 0)
        if header_idx >= len(all_rows): 
            header_idx = 0
        headers = all_rows[header_idx]
        
        # 3. Format Data
        sync_timestamp = int(time.time() * 1000) # JS compatible timestamp
        
        clean_brand_name = brandName.strip()

        # Delete old brand data in D1 before inserting new (wrapped in try/except for safety)
        try:
            execute_d1_query("DELETE FROM products WHERE brand_id = ?", [clean_brand_name])
        except Exception as e:
            print(f"Delete operation warning: {e}")
        
        valid_products = []
        # Skip rows up to and including the AI-identified header row
        for row in all_rows[header_idx + 1:]:
            if len(row) < 3: continue
                
            attrs = {}
            if mapping.get("attribute_indices"):
                for idx in mapping["attribute_indices"]:
                    try:
                        idx = int(idx)
                        if idx != -1 and idx < len(row) and idx < len(headers):
                            clean_header = ''.join(c for c in headers[idx] if c.isalnum() or c.isspace()).strip()
                            # Prevent core fields from accidentally showing up as spec badges
                            if clean_header and row[idx] and idx not in [mapping.get("mrp_index"), mapping.get("list_price_ex_gst_index"), mapping.get("list_price_inc_gst_index"), mapping.get("model_name_index")]:
                                attrs[clean_header] = row[idx]
                    except (ValueError, TypeError):
                        continue
                            
            def get_val(key):
                idx = mapping.get(key, -1)
                return row[idx] if idx != -1 and idx < len(row) else ""

            ex_gst = clean_price(get_val("list_price_ex_gst_index"))
            inc_gst = clean_price(get_val("list_price_inc_gst_index"))
            mrp = clean_price(get_val("mrp_index"))
            model_name = str(get_val("model_name_index")).strip()
            
            # Reverted back to your exact working condition logic
            if ex_gst > 0 and inc_gst > 0 and model_name:
                # Include the attributes dictionary in the ID generation so variants have distinct IDs
                product_id = get_deterministic_id(clean_brand_name, model_name, attrs)
                # Append row data as a list of params
                valid_products.append([
                    product_id, clean_brand_name, model_name, mrp, 
                    ex_gst, inc_gst, json.dumps(attrs), sync_timestamp
                ])
                
        # 4. Push to Cloudflare D1 via Multi-Row Batch Inserts
        chunk_size = 12
        for i in range(0, len(valid_products), chunk_size):
            chunk = valid_products[i:i + chunk_size]
            
            placeholders = ",".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(chunk))
            # Flatten the nested list of parameters
            params = [item for row in chunk for item in row]
            
            sql = f"""INSERT OR REPLACE INTO products 
                      (id, brand_id, model_name, mrp, list_price_ex_gst, list_price_inc_gst, attributes, updated_at) 
                      VALUES {placeholders}"""
            execute_d1_query(sql, params)
            
        return {
            "status": "success", 
            "message": f"Successfully processed and synced {len(valid_products)} products to D1."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)

# ==========================================
# ENDPOINT 2: DELETE BRAND (Used heavily by Edge logic)
# ==========================================
@app.delete("/api/brand/{brand_name}")
async def delete_brand(brand_name: str):
    try:
        clean_brand_name = brand_name.strip()
        execute_d1_query("DELETE FROM products WHERE brand_id = ?", [clean_brand_name])
        return {"status": "success", "message": f"Deleted all data for brand: {clean_brand_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
