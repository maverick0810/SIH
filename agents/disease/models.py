# server/agents/disease/models.py
"""
Pydantic models for disease detection agent
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DiseaseDetectionRequest(BaseModel):
    crop_type: Optional[str] = Field(None, description="Type of crop/plant (e.g., wheat, rice, tomato)")
    location: Optional[str] = Field(None, description="Location where plant is grown")
    symptoms_description: Optional[str] = Field(None, description="Description of visible symptoms")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image of the plant")

class TreatmentMethod(BaseModel):
    method: str = Field(..., description="Treatment method name")
    description: str = Field(..., description="Detailed description of the treatment")
    application: str = Field(..., description="How to apply this treatment")
    frequency: str = Field(..., description="How often to apply")
    cost_estimate: Optional[str] = Field(None, description="Estimated cost range")

class PreventionMeasure(BaseModel):
    measure: str = Field(..., description="Prevention measure")
    description: str = Field(..., description="How this helps prevent disease")
    timing: str = Field(..., description="When to implement this measure")

class DiseaseInfo(BaseModel):
    disease_name: str = Field(..., description="Name of the identified disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    severity: str = Field(..., description="Severity level: mild, moderate, severe")
    description: str = Field(..., description="Description of the disease")
    causes: List[str] = Field(..., description="What causes this disease")
    symptoms: List[str] = Field(..., description="Common symptoms to look for")
    affected_parts: List[str] = Field(..., description="Which parts of plant are affected")
    spread_mechanism: str = Field(..., description="How the disease spreads")
    favorable_conditions: List[str] = Field(..., description="Conditions that favor disease development")

class CropIdentification(BaseModel):
    crop_type: str = Field(..., description="Identified crop type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in crop identification")
    growth_stage: str = Field(..., description="Current growth stage")
    visual_indicators: List[str] = Field(..., description="Visual clues used for identification")

class DiseaseDetectionResult(BaseModel):
    detected_diseases: List[DiseaseInfo] = Field(..., description="List of possible diseases")
    primary_disease: DiseaseInfo = Field(..., description="Most likely disease")
    crop_identification: Optional[CropIdentification] = Field(None, description="Automatically identified crop information")
    treatment_methods: List[TreatmentMethod] = Field(..., description="Treatment options")
    prevention_measures: List[PreventionMeasure] = Field(..., description="Prevention strategies")
    immediate_actions: List[str] = Field(..., description="Immediate steps to take")
    when_to_seek_expert: str = Field(..., description="When to consult agricultural expert")
    overall_plant_health: Optional[str] = Field(None, description="Overall health assessment")
    additional_observations: Optional[str] = Field(None, description="Additional findings")

class DiseaseDetectionResponse(BaseModel):
    success: bool
    data: Optional[DiseaseDetectionResult] = None
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None