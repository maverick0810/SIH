# server/agents/disease/service.py
"""
Disease detection service using Google Generative AI for image analysis
"""
import base64
import io
import json
import os
from typing import Dict, Any, Optional
from PIL import Image
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from agents.disease.models import (
    DiseaseDetectionRequest, DiseaseDetectionResult, DiseaseInfo, 
    TreatmentMethod, PreventionMeasure
)

logger = logging.getLogger(__name__)

class DiseaseDetectionService:
    """Service for detecting plant diseases using Google Generative AI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.google_api_key:
            logger.warning("No GOOGLE_API_KEY found - disease detection will not work")
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=False
            )
            logger.info("Disease detection service initialized with Google Generative AI")
    
    def validate_image(self, image_base64: str) -> bool:
        """Validate and check image format"""
        try:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            
            # Try to open with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'WEBP']:
                return False
            
            # Check size (should be reasonable)
            width, height = image.size
            if width < 50 or height < 50 or width > 4096 or height > 4096:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def analyze_plant_image(self, request: DiseaseDetectionRequest) -> DiseaseDetectionResult:
        """Analyze plant image for disease detection using Google Generative AI"""
        
        if not self.llm:
            raise RuntimeError("Google API key not configured - cannot perform disease detection")
        
        # Validate image if provided
        if request.image_base64:
            if not self.validate_image(request.image_base64):
                raise ValueError("Invalid image format or size")
        
        # Prepare system prompt
        system_prompt = self._get_system_prompt()
        
        # Prepare human message with context
        human_message_content = self._prepare_analysis_request(request)
        
        try:
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message_content)
            ]
            
            # Add image to the message if provided
            if request.image_base64:
                # Clean base64 string
                image_data = request.image_base64
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Create message with image
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": human_message_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ])
                ]
            
            # Get AI response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse JSON response
            parsed_result = self._parse_ai_response(response_text)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Disease detection analysis failed: {e}")
            raise RuntimeError(f"Disease analysis failed: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for disease detection"""
        return """You are an expert plant pathologist and agricultural advisor. Your task is to analyze plant images to automatically identify the crop type and any diseases present.

When analyzing an image, you must:
1. FIRST identify the crop/plant type from visual characteristics
2. THEN analyze for diseases, pests, or health issues
3. Consider growth stage, leaf patterns, plant structure
4. Look for visual symptoms (spots, discoloration, wilting, pests, etc.)
5. Assess severity and provide confidence scores

IMPORTANT: Provide your response as a valid JSON object with this exact structure:
{
  "detected_diseases": [
    {
      "disease_name": "string",
      "confidence": 0.0-1.0,
      "severity": "mild|moderate|severe",
      "description": "string",
      "causes": ["cause1", "cause2"],
      "symptoms": ["symptom1", "symptom2"],
      "affected_parts": ["leaves", "stem", "roots"],
      "spread_mechanism": "string",
      "favorable_conditions": ["condition1", "condition2"]
    }
  ],
  "primary_disease": {
    // Same structure as above for most likely disease - if no disease detected, use "Healthy Plant"
  },
  "crop_identification": {
    "crop_type": "string (e.g., wheat, rice, tomato, cotton)",
    "confidence": 0.0-1.0,
    "growth_stage": "seedling|vegetative|flowering|fruiting|mature",
    "visual_indicators": ["leaf_shape", "plant_structure", "etc"]
  },
  "treatment_methods": [
    {
      "method": "string",
      "description": "string",
      "application": "string",
      "frequency": "string",
      "cost_estimate": "string"
    }
  ],
  "prevention_measures": [
    {
      "measure": "string",
      "description": "string", 
      "timing": "string"
    }
  ],
  "immediate_actions": ["action1", "action2"],
  "when_to_seek_expert": "string",
  "overall_plant_health": "excellent|good|fair|poor",
  "additional_observations": "any other relevant findings"
}

Focus on:
- Accurate crop identification from image features
- Disease detection with confidence levels
- Practical, actionable advice for farmers
- Both organic and chemical treatment options
- If the plant appears healthy, still provide the crop identification and general care recommendations"""
    
    def _prepare_analysis_request(self, request: DiseaseDetectionRequest) -> str:
        """Prepare the analysis request text"""
        context_parts = []
        
        if request.crop_type:
            context_parts.append(f"Crop type: {request.crop_type}")
        
        if request.location:
            context_parts.append(f"Location: {request.location}")
        
        if request.symptoms_description:
            context_parts.append(f"Observed symptoms: {request.symptoms_description}")
        
        context = "\n".join(context_parts) if context_parts else "No additional context provided"
        
        request_text = f"""Please analyze this plant for diseases and provide treatment recommendations.

Context:
{context}

{"Image is attached for visual analysis." if request.image_base64 else "No image provided - analyze based on context only."}

Provide a comprehensive disease analysis including identification, treatment methods, and prevention measures in the specified JSON format."""
        
        return request_text
    
    def _parse_ai_response(self, response_text: str) -> DiseaseDetectionResult:
        """Parse AI response and convert to structured format"""
        try:
            # Clean response text (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            parsed_data = json.loads(cleaned_text)
            
            # Convert to Pydantic models
            detected_diseases = [DiseaseInfo(**disease) for disease in parsed_data.get("detected_diseases", [])]
            primary_disease = DiseaseInfo(**parsed_data["primary_disease"])
            treatment_methods = [TreatmentMethod(**method) for method in parsed_data.get("treatment_methods", [])]
            prevention_measures = [PreventionMeasure(**measure) for measure in parsed_data.get("prevention_measures", [])]
            
            # Handle crop identification (new feature)
            crop_identification = None
            if "crop_identification" in parsed_data:
                from agents.disease.models import CropIdentification
                crop_identification = CropIdentification(**parsed_data["crop_identification"])
            
            result = DiseaseDetectionResult(
                detected_diseases=detected_diseases,
                primary_disease=primary_disease,
                crop_identification=crop_identification,
                treatment_methods=treatment_methods,
                prevention_measures=prevention_measures,
                immediate_actions=parsed_data.get("immediate_actions", []),
                when_to_seek_expert=parsed_data.get("when_to_seek_expert", "Consult agricultural expert if symptoms worsen"),
                overall_plant_health=parsed_data.get("overall_plant_health"),
                additional_observations=parsed_data.get("additional_observations")
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            raise ValueError("AI response was not valid JSON")
        except Exception as e:
            logger.error(f"Failed to convert AI response to structured format: {e}")
            raise ValueError(f"Failed to process AI response: {str(e)}")
    
    def get_common_diseases(self, crop_type: str) -> Dict[str, Any]:
        """Get common diseases for a specific crop type"""
        common_diseases = {
            "wheat": [
                "Wheat Rust", "Powdery Mildew", "Septoria Leaf Blotch", "Fusarium Head Blight"
            ],
            "rice": [
                "Bacterial Leaf Blight", "Rice Blast", "Brown Spot", "Sheath Blight"
            ],
            "cotton": [
                "Cotton Bollworm", "Verticillium Wilt", "Fusarium Wilt", "Bacterial Blight"
            ],
            "tomato": [
                "Late Blight", "Early Blight", "Bacterial Spot", "Fusarium Wilt", "Mosaic Virus"
            ],
            "potato": [
                "Late Blight", "Early Blight", "Common Scab", "Potato Virus Y"
            ],
            "maize": [
                "Northern Corn Leaf Blight", "Gray Leaf Spot", "Common Rust", "Smut"
            ]
        }
        
        return {
            "crop": crop_type,
            "common_diseases": common_diseases.get(crop_type.lower(), []),
            "note": "These are common diseases for this crop type"
        }