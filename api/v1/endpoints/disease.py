# server/api/v1/endpoints/disease.py - UPDATED VERSION
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import base64
import sys
import os

# Add the server directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.base import agent_registry
from agents.disease.models import DiseaseDetectionRequest

router = APIRouter()

@router.post("/analyze-image")
async def analyze_plant_image(
    image: UploadFile = File(..., description="Plant image - AI will detect crop type and diseases automatically")
):
    """
    Smart plant disease detection - just upload an image!
    
    The AI will automatically:
    - Identify the crop type from the image
    - Detect any diseases present
    - Provide treatment recommendations
    - Suggest prevention measures
    
    No need to specify crop type or symptoms - the AI figures it all out!
    """
    try:
        # Get the disease detection agent
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        # Validate file type
        if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Supported formats: JPEG, PNG, WEBP"
            )
        
        # Check file size (max 5MB)
        content = await image.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=400, detail="Image file too large. Maximum size: 5MB")
        
        # Convert to base64
        image_base64 = base64.b64encode(content).decode('utf-8')
        
        # Create request with enhanced prompt for automatic detection
        request = DiseaseDetectionRequest(
            crop_type=None,  # Let AI detect this
            location=None,   # Let AI analyze from image context
            symptoms_description="Analyze this plant image for crop identification and disease detection",
            image_base64=image_base64
        )
        
        # Execute the request
        response = await disease_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing plant image: {str(e)}")

@router.post("/detect")
async def detect_disease(
    crop_type: Optional[str] = Form(None, description="Type of crop (e.g., wheat, rice, tomato)"),
    location: Optional[str] = Form(None, description="Location where plant is grown"),
    symptoms_description: Optional[str] = Form(None, description="Description of visible symptoms"),
    image: Optional[UploadFile] = File(None, description="Plant image file (JPEG/PNG/WEBP)")
):
    """
    Advanced disease detection with optional manual inputs
    
    Upload an image and optionally provide additional context.
    The more information you provide, the more accurate the analysis.
    """
    try:
        # Get the disease detection agent
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        # Validate input
        if not image and not symptoms_description:
            raise HTTPException(
                status_code=400, 
                detail="Either image file or symptoms description must be provided"
            )
        
        # Process image if provided
        image_base64 = None
        if image:
            # Validate file type
            if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image format. Supported formats: JPEG, PNG, WEBP"
                )
            
            # Check file size (max 5MB)
            content = await image.read()
            if len(content) > 5 * 1024 * 1024:  # 5MB
                raise HTTPException(status_code=400, detail="Image file too large. Maximum size: 5MB")
            
            # Convert to base64
            image_base64 = base64.b64encode(content).decode('utf-8')
        
        # Create request
        request = DiseaseDetectionRequest(
            crop_type=crop_type.strip() if crop_type else None,
            location=location.strip() if location else None,
            symptoms_description=symptoms_description.strip() if symptoms_description else None,
            image_base64=image_base64
        )
        
        # Execute the request
        response = await disease_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing disease detection: {str(e)}")

@router.post("/detect-base64")
async def detect_disease_base64(
    crop_type: Optional[str] = None,
    location: Optional[str] = None,
    symptoms_description: Optional[str] = None,
    image_base64: Optional[str] = None
):
    """
    Detect plant diseases using base64 encoded image
    
    Alternative endpoint for cases where you have base64 encoded image data.
    """
    try:
        # Get the disease detection agent
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        # Validate input
        if not image_base64 and not symptoms_description:
            raise HTTPException(
                status_code=400,
                detail="Either image_base64 or symptoms_description must be provided"
            )
        
        # Create request
        request = DiseaseDetectionRequest(
            crop_type=crop_type.strip() if crop_type else None,
            location=location.strip() if location else None,
            symptoms_description=symptoms_description.strip() if symptoms_description else None,
            image_base64=image_base64
        )
        
        # Execute the request
        response = await disease_agent.execute(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing disease detection: {str(e)}")

@router.get("/crops")
async def get_supported_crops():
    """Get crops supported for disease detection"""
    try:
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        crops = await disease_agent.get_supported_crops()
        return {
            "success": True,
            "crops": crops,
            "note": "These crops have optimized disease detection models"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting supported crops: {str(e)}")

@router.get("/diseases/{crop_type}")
async def get_common_diseases(crop_type: str):
    """Get common diseases for a specific crop type"""
    try:
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        diseases = await disease_agent.get_common_diseases(crop_type)
        return {
            "success": True,
            **diseases
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting diseases for {crop_type}: {str(e)}")

@router.get("/photo-guidelines")
async def get_photo_guidelines():
    """Get guidelines for taking good disease detection photos"""
    try:
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            raise HTTPException(status_code=500, detail="Disease detection agent not available")
        
        guidelines = await disease_agent.get_image_guidelines()
        return {
            "success": True,
            **guidelines
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting photo guidelines: {str(e)}")

@router.get("/health")
async def disease_health():
    """Check disease detection agent health"""
    try:
        disease_agent = agent_registry.get("disease")
        if not disease_agent:
            return {"status": "unhealthy", "error": "Disease detection agent not available"}
        
        health = await disease_agent.health_check()
        
        # Add Google API key status
        google_api_available = bool(os.getenv('GOOGLE_API_KEY'))
        health["google_api_configured"] = google_api_available
        
        if not google_api_available:
            health["warnings"] = ["GOOGLE_API_KEY not configured - limited functionality"]
        
        return health
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}