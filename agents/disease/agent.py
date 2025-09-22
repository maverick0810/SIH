# server/agents/disease/agent.py
"""
Plant disease detection agent using Google Generative AI
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Add server directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.base import BaseAgent
from agents.disease.models import (
    DiseaseDetectionRequest, DiseaseDetectionResponse, DiseaseDetectionResult,
    DiseaseInfo, TreatmentMethod, PreventionMeasure
)
from agents.disease.service import DiseaseDetectionService
from core.exceptions import AgentError, AgentConfigError

class DiseaseDetectionAgent(BaseAgent[DiseaseDetectionRequest, DiseaseDetectionResponse]):
    """
    Plant disease detection agent using Google Generative AI
    
    Features:
    - Image-based disease analysis using Google Gemini Vision
    - Comprehensive disease identification
    - Treatment recommendations (organic and chemical)
    - Prevention strategies
    - Immediate action guidance
    - Confidence scoring
    """
    
    def __init__(self):
        super().__init__("disease")
        
        # Check for Google API key
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            self.logger.warning("No GOOGLE_API_KEY found - disease detection will have limited functionality")
        else:
            self.logger.info("Disease detection agent initialized with Google API")
        
        self.service = DiseaseDetectionService(config=self.config)
        self.logger.info("Disease detection agent initialized")
    
    def _validate_config(self) -> None:
        """Validate disease detection agent configuration"""
        # Check if Google API key is available
        if not os.getenv('GOOGLE_API_KEY'):
            self.logger.warning("GOOGLE_API_KEY not found - add to .env file for full functionality")
        
        # Optional config validation
        optional_config = ["max_image_size_mb", "supported_formats", "confidence_threshold"]
        for key in optional_config:
            if key not in self.config:
                self.logger.debug(f"Optional config {key} not set, using defaults")
    
    async def process_request(self, request: DiseaseDetectionRequest) -> DiseaseDetectionResponse:
        """Process disease detection request"""
        
        self.logger.info(f"Processing disease detection request for crop: {request.crop_type}")
        
        try:
            # Validate request
            if not request.image_base64 and not request.symptoms_description:
                raise ValueError("Either image or symptoms description must be provided")
            
            # Analyze using the service
            result = await self._analyze_disease(request)
            
            # Create response
            message = self._generate_response_message(result)
            
            response = DiseaseDetectionResponse(
                success=True,
                data=result,
                message=message,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "request_params": {
                        "has_image": bool(request.image_base64),
                        "has_symptoms": bool(request.symptoms_description),
                        "crop_type": request.crop_type,
                        "location": request.location
                    },
                    "analysis_method": "google_generative_ai",
                    "diseases_detected": len(result.detected_diseases),
                    "primary_disease_confidence": result.primary_disease.confidence
                }
            )
            
            self.logger.info(f"Disease analysis completed. Primary disease: {result.primary_disease.disease_name} (confidence: {result.primary_disease.confidence:.2f})")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing disease detection request: {e}")
            raise AgentError(f"Failed to process disease detection request: {e}")
    
    async def _analyze_disease(self, request: DiseaseDetectionRequest) -> DiseaseDetectionResult:
        """Analyze disease using the service"""
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.service.analyze_plant_image,
            request
        )
    
    def _generate_response_message(self, result: DiseaseDetectionResult) -> str:
        """Generate response message based on analysis results"""
        primary_disease = result.primary_disease
        confidence_level = "high" if primary_disease.confidence >= 0.8 else "medium" if primary_disease.confidence >= 0.6 else "low"
        
        # Include crop identification in message if available
        crop_info = ""
        if result.crop_identification:
            crop_confidence = "high" if result.crop_identification.confidence >= 0.8 else "medium" if result.crop_identification.confidence >= 0.6 else "low"
            crop_info = f"Identified as {result.crop_identification.crop_type} ({crop_confidence} confidence). "
        
        # Health and urgency assessment
        if primary_disease.disease_name.lower() in ["healthy plant", "no disease detected", "healthy"]:
            urgency = "Plant appears healthy. Continue regular care."
        elif primary_disease.severity == "severe":
            urgency = "Immediate action required!"
        elif primary_disease.severity == "moderate":
            urgency = "Treatment recommended soon."
        else:
            urgency = "Monitor and apply preventive measures."
        
        return f"{crop_info}Detected {primary_disease.disease_name} with {confidence_level} confidence ({primary_disease.confidence:.2f}). Severity: {primary_disease.severity}. {urgency}"
    
    def get_fallback_response(self, request: DiseaseDetectionRequest, error: Exception) -> DiseaseDetectionResponse:
        """Get fallback response when agent fails"""
        
        # Generate basic fallback result based on crop type
        fallback_disease = DiseaseInfo(
            disease_name="General Plant Stress",
            confidence=0.3,
            severity="unknown",
            description="Unable to perform detailed analysis. General plant health recommendations provided.",
            causes=["Environmental stress", "Nutrient deficiency", "Possible disease"],
            symptoms=["Visible plant distress"],
            affected_parts=["Multiple plant parts"],
            spread_mechanism="Variable",
            favorable_conditions=["Stress conditions"]
        )
        
        fallback_result = DiseaseDetectionResult(
            detected_diseases=[fallback_disease],
            primary_disease=fallback_disease,
            treatment_methods=[
                TreatmentMethod(
                    method="General Plant Care",
                    description="Ensure proper watering, nutrition, and environmental conditions",
                    application="Follow standard agricultural practices for your crop",
                    frequency="As needed",
                    cost_estimate="Variable"
                )
            ],
            prevention_measures=[
                PreventionMeasure(
                    measure="Regular Monitoring",
                    description="Check plants regularly for signs of disease or stress",
                    timing="Daily during growing season"
                )
            ],
            immediate_actions=[
                "Consult local agricultural extension officer",
                "Take clear photos of affected plants",
                "Remove severely affected plant parts if safe to do so"
            ],
            when_to_seek_expert="Immediately - automated analysis failed"
        )
        
        return DiseaseDetectionResponse(
            success=False,
            data=fallback_result,
            message=f"Disease analysis failed: {str(error)}. General recommendations provided.",
            timestamp=datetime.now().isoformat(),
            metadata={"fallback": True, "error": str(error)}
        )
    
    async def get_supported_crops(self) -> List[Dict[str, Any]]:
        """Get crops supported for disease detection"""
        crops = [
            {
                "name": "Wheat",
                "common_diseases": ["Wheat Rust", "Powdery Mildew", "Septoria Leaf Blotch"],
                "detection_accuracy": "high"
            },
            {
                "name": "Rice", 
                "common_diseases": ["Bacterial Leaf Blight", "Rice Blast", "Brown Spot"],
                "detection_accuracy": "high"
            },
            {
                "name": "Cotton",
                "common_diseases": ["Cotton Bollworm", "Verticillium Wilt", "Fusarium Wilt"],
                "detection_accuracy": "medium"
            },
            {
                "name": "Tomato",
                "common_diseases": ["Late Blight", "Early Blight", "Bacterial Spot"],
                "detection_accuracy": "high"
            },
            {
                "name": "Potato",
                "common_diseases": ["Late Blight", "Early Blight", "Common Scab"],
                "detection_accuracy": "high"
            },
            {
                "name": "Maize",
                "common_diseases": ["Northern Corn Leaf Blight", "Gray Leaf Spot"],
                "detection_accuracy": "medium"
            }
        ]
        return crops
    
    async def get_common_diseases(self, crop_type: str) -> Dict[str, Any]:
        """Get common diseases for a specific crop"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.service.get_common_diseases,
            crop_type
        )
    
    async def get_image_guidelines(self) -> Dict[str, Any]:
        """Get guidelines for taking good disease detection photos"""
        return {
            "photo_tips": [
                "Take photos in good natural lighting",
                "Focus on affected areas clearly",
                "Include both affected and healthy parts for comparison",
                "Take multiple angles if possible",
                "Avoid blurry or very dark images"
            ],
            "technical_requirements": {
                "formats": ["JPEG", "PNG", "WEBP"],
                "max_size": "5MB",
                "min_resolution": "100x100 pixels",
                "max_resolution": "4096x4096 pixels"
            },
            "what_to_include": [
                "Close-up of symptoms",
                "Whole plant context",
                "Affected leaves, stems, or fruits",
                "Any visible pests or fungal growth"
            ],
            "avoid": [
                "Very blurry images",
                "Poor lighting conditions",
                "Images with only soil or background",
                "Overly small affected areas"
            ]
        }