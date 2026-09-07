"""
schemas.py
==========
Pydantic models = the "contract" of the API.

`EmployeeInput`  -> what the frontend form must send (one employee's raw data,
                    exactly as it appears in the original CSV columns).
`PredictionResponse` -> what the backend sends back.

Using `Literal[...]` for categorical fields means FastAPI will REJECT a
request automatically (with a clear 422 error) if, say, "Department" is
misspelled — we don't have to write that validation ourselves.
"""

from typing import Literal

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    # --- Demographics -------------------------------------------------
    Age: int = Field(..., ge=18, le=60, description="Employee age")
    Gender: Literal["Male", "Female"]
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    DistanceFromHome: int = Field(..., ge=0, le=30, description="Miles from home to office")
    Education: int = Field(..., ge=1, le=5, description="1=Below College ... 5=Doctor")
    EducationField: Literal[
        "Life Sciences", "Medical", "Marketing", "Technical Degree",
        "Human Resources", "Other",
    ]

    # --- Job info -------------------------------------------------------
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources",
    ]
    JobLevel: int = Field(..., ge=1, le=5)
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    OverTime: Literal["Yes", "No"]
    NumCompaniesWorked: int = Field(..., ge=0, le=9)
    TotalWorkingYears: int = Field(..., ge=0, le=40)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsInCurrentRole: int = Field(..., ge=0, le=18)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15)
    YearsWithCurrManager: int = Field(..., ge=0, le=17)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)

    # --- Compensation ----------------------------------------------------
    MonthlyIncome: int = Field(..., ge=1000, le=20000)
    DailyRate: int = Field(..., ge=100, le=1500)
    HourlyRate: int = Field(..., ge=30, le=100)
    MonthlyRate: int = Field(..., ge=2000, le=27000)
    PercentSalaryHike: int = Field(..., ge=11, le=25)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    PerformanceRating: int = Field(..., ge=3, le=4)

    # --- Satisfaction / work-life scores (1=Low ... 4=Very High) --------
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    JobInvolvement: int = Field(..., ge=1, le=4)

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 34, "Gender": "Male", "MaritalStatus": "Single",
                "DistanceFromHome": 8, "Education": 3, "EducationField": "Life Sciences",
                "Department": "Sales", "JobRole": "Sales Executive", "JobLevel": 2,
                "BusinessTravel": "Travel_Rarely", "OverTime": "Yes",
                "NumCompaniesWorked": 3, "TotalWorkingYears": 8, "YearsAtCompany": 4,
                "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 2, "TrainingTimesLastYear": 2,
                "MonthlyIncome": 4500, "DailyRate": 800, "HourlyRate": 65,
                "MonthlyRate": 15000, "PercentSalaryHike": 14, "StockOptionLevel": 0,
                "PerformanceRating": 3, "EnvironmentSatisfaction": 2,
                "JobSatisfaction": 2, "RelationshipSatisfaction": 3,
                "WorkLifeBalance": 2, "JobInvolvement": 3,
            }
        }


class FactorContribution(BaseModel):
    feature: str
    contribution: float   # positive = pushes risk up, negative = pushes it down


class PredictionResponse(BaseModel):
    attrition_probability: float
    risk_tier: Literal["High Risk", "Medium Risk", "Low Risk"]
    decision_threshold_used: float
    predicted_attrition: bool  # probability >= threshold
    top_risk_factors: list[FactorContribution]   # factors pushing risk UP
    top_protective_factors: list[FactorContribution]  # factors pushing risk DOWN
