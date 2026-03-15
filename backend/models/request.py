"""
BeanHealth CLR Tool — API Request Schema
=========================================

Pydantic v2 models for validating incoming requests to POST /analyse.

Request format: multipart/form-data
    image        : UploadFile  — JPEG or PNG image (required)
    patient_name : str         — patient's name (required, 1–100 chars)
    patient_age  : int         — patient's age in years (required, 1–120)

FastAPI reads the image bytes from the upload and the text fields from the
form body. The image is not part of the Pydantic model (UploadFile is
handled directly by FastAPI), but age and name are validated here.
"""

from pydantic import BaseModel, Field, field_validator


class AnalyseRequest(BaseModel):
    """
    Form-field portion of the /analyse request.

    Note: the 'image' file is declared as a FastAPI parameter directly
    on the route (as UploadFile), NOT in this model, because Pydantic v2
    does not support file uploads inside BaseModel.
    """

    patient_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Patient's full name",
        examples=["Emma Wilson"],
    )

    patient_age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Patient's age in years",
        examples=[5],
    )

    @field_validator("patient_name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("patient_name must not be blank or whitespace-only")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "patient_name": "Emma Wilson",
                    "patient_age": 5,
                }
            ]
        }
    }
