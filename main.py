from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import uvicorn
import os
import shutil
from datetime import datetime
import uuid
from cropper import process_image
from m_ocr import MOCRSystem
from ultralytics import YOLO

app = FastAPI(title="ID Card Processing API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Create directories for temporary and cropped images
UPLOAD_DIR = "temp_uploads"
CROPPED_DIR = "cropped_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

# Serve cropped_images as static files
app.mount("/cropped_images", StaticFiles(directory="cropped_images"), name="cropped_images")

# Initialize models
crop_model = YOLO("model/detect_4goc/cropper.pt")
id_system = MOCRSystem()

class FilePath(BaseModel):
    file_path: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"message": "Upload successful", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crop")
async def crop_image(file_data: FilePath):
    try:
        print(f"Received file path: {file_data.file_path}")
        # Check the file exists
        if not os.path.exists(file_data.file_path):
            print(f"File not found at path: {file_data.file_path}")
            raise HTTPException(status_code=404, detail=f"File not found at path: {file_data.file_path}")
        
        # Crop image using cropper
        cropped_filename = process_image(file_data.file_path, CROPPED_DIR, crop_model)
        if not cropped_filename:
            raise HTTPException(status_code=400, detail="Unable to crop image")
        
        return {
            "status": "success",
            "message": "Cropping successful",
            "cropped_image_path": cropped_filename
        }
    except Exception as e:
        print(f"Error in crop_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_info(file_data: FilePath):
    try:
        # Build full path to cropped file
        full_path = os.path.join(CROPPED_DIR, file_data.file_path)
        print(f"Extracting info from file: {full_path}")
        
        # Check the file exists
        if not os.path.exists(full_path):
            print(f"File not found at path: {full_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Process image with m_ocr
        try:
            result = id_system.process_id_card(full_path)
            if not result:
                print("No result returned from process_id_card")
                raise HTTPException(status_code=400, detail="Unable to process ID information")
        except Exception as e:
            print(f"Error in process_id_card: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
        
        return {
            "status": "success",
            "id_type": result['id_type'],
            "extracted_info": result['parsed_info']
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error in extract_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 