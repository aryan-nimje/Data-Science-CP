from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import pandas as pd

from backend.data_engine.investigator import DatasetInvestigator
from backend.data_engine.target_detector import TargetDetector
from backend.data_engine.preprocessor import AutomaticPreprocessor
from backend.data_engine.preprocessing_pipeline import PreprocessingPipeline
from backend.ml_engine.model_trainer import ModelTrainer
from backend.ml_engine.prediction_engine import PredictionEngine
from backend.ml_engine.explainability import ExplainabilityEngine
from backend.ml_engine.error_analyzer import ErrorAnalyzer
from backend.ml_engine.what_if import WhatIfEngine
from backend.report import ModelInvestigationReport
from backend.ml_engine.feature_importance import FeatureImportanceEngine
from pydantic import BaseModel

class PredictionRequest(BaseModel):

    Age: float
    Income: float
    City: str
    Experience: float

class WhatIfRequest(BaseModel):
    filename: str
    row: dict
    feature: str
    new_value: object

class ReportRequest(BaseModel):
    filename: str


app = FastAPI(
    title="AI Data Scientist",
    description="Automated Dataset Investigation and Machine Learning Platform",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "datasets/uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.get("/")
def root():

    return {
        "message": "AI Data Scientist API is running",
        "status": "online"
    }


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are currently supported."
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:

        # ==================================================
        # SAVE DATASET
        # ==================================================

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ==================================================
        # DATASET INVESTIGATION
        # ==================================================

        investigator = DatasetInvestigator(
            file_path
        )

        investigation = (
            investigator.investigate()
        )

        # ==================================================
        # TARGET DETECTION
        # ==================================================

        df = pd.read_csv(
            file_path
        )

        target_detector = TargetDetector(
            df
        )

        target_analysis = (
            target_detector.analyze()
        )

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return {

            "success": True,

            "filename":
                file.filename,

            "investigation":
                investigation,

            "target_analysis":
                target_analysis
        }

    except Exception as error:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

@app.post("/train")
async def train_model(
    filename: str
):

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    if not os.path.exists(file_path):

        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Upload the dataset first."
        )

    try:

        # ==================================================
        # LOAD DATASET
        # ==================================================

        df = pd.read_csv(
            file_path
        )

        # ==================================================
        # TARGET DETECTION
        # ==================================================

        detector = TargetDetector(
            df
        )

        target_result = detector.analyze()

        target = (
            target_result[
                "selected_target"
            ]
        )

        if target is None:

            raise HTTPException(
                status_code=400,
                detail="No suitable target column detected."
            )

        problem_type = (
            target_result[
                "problem"
            ][
                "problem_type"
            ]
        )

        # ==================================================
        # AUTOMATIC PREPROCESSING PLAN
        # ==================================================

        planner = AutomaticPreprocessor(
            df,
            target
        )

        plan = (
            planner.generate_plan()
        )

        numerical_features = (
            plan["features"]
            ["numerical"]
        )

        categorical_features = (
            plan["features"]
            ["categorical"]
        )

        removed_columns = (
            plan["removed_columns"]
            ["id_like"]
            +
            plan["removed_columns"]
            ["constant"]
        )

        # ==================================================
        # BUILD PREPROCESSING PIPELINE
        # ==================================================

        processor = PreprocessingPipeline(
            df,
            target,
            numerical_features,
            categorical_features,
            removed_columns
        )

        preprocessing_pipeline = (
            processor.build()
        )

        X, y = (
            processor.prepare_data()
        )

        # ==================================================
        # MODEL TRAINING
        # ==================================================

        trainer = ModelTrainer(
            preprocessing_pipeline,
            X,
            y,
            problem_type
        )

        training_result = (
            trainer.train()
        )

        # ==================================================
        # SAVE BEST MODEL
        # ==================================================

        model_path = (
            trainer.save_model()
        )

        # ==================================================
        # RESPONSE
        # ==================================================

        return {

            "success": True,

            "dataset":
                filename,

            "target":
                target,

            "problem_type":
                problem_type,

            "preprocessing_plan":
                plan,

            "training":
                training_result,

            "model_path":
                model_path
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
    
@app.post("/predict")
async def predict(
    request: PredictionRequest
):

    model_path = "models/best_model_pipeline.pkl"

    if not os.path.exists(model_path):

        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Train the model first."
        )

    try:

        # ==============================================
        # LOAD TRAINED MODEL
        # ==============================================

        prediction_engine = PredictionEngine(
            model_path=model_path
        )

        # ==============================================
        # PREPARE INPUT
        # ==============================================

        input_data = {
            "Age": request.Age,
            "Income": request.Income,
            "City": request.City,
            "Experience": request.Experience
        }

        # ==============================================
        # PREDICT
        # ==============================================

        result = prediction_engine.predict_single(
            input_data
        )

        # ==============================================
        # RETURN RESULT
        # ==============================================

        return {
            "success": True,
            "input": input_data,
            "prediction": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/explain")
async def explain_prediction(
    request: PredictionRequest
):

    model_path = "models/best_model_pipeline.pkl"

    if not os.path.exists(model_path):

        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Train the model first."
        )

    try:

        # ==============================================
        # PREPARE INPUT
        # ==============================================

        input_data = {
            "Age": request.Age,
            "Income": request.Income,
            "City": request.City,
            "Experience": request.Experience
        }

        # ==============================================
        # PREDICTION
        # ==============================================

        prediction_engine = PredictionEngine(
            model_path=model_path
        )

        prediction_result = (
            prediction_engine.predict_single(
                input_data
            )
        )

        # ==============================================
        # SHAP EXPLANATION
        # ==============================================

        explainability_engine = ExplainabilityEngine(
            model_path=model_path,
            background_path="datasets/uploads/test.csv"
        )

        shap_result = (
            explainability_engine.explain_prediction(
                input_data
            )
        )

        # ==============================================
        # RETURN RESULT
        # ==============================================

        return {

            "success": True,

            "input":
                input_data,

            "prediction":
                prediction_result,

            "explanation": {

                "model":
                    shap_result["model"],

                "base_value":
                    shap_result["base_value"],

                "positive_factors":
                    shap_result[
                        "positive_factors"
                    ][:5],

                "negative_factors":
                    shap_result[
                        "negative_factors"
                    ][:5]
            }
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/errors")
async def analyze_errors(
    filename: str
):

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    if not os.path.exists(file_path):

        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Upload the dataset first."
        )

    model_path = "models/best_model_pipeline.pkl"

    if not os.path.exists(model_path):

        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Train the model first."
        )

    try:

        # ==============================================
        # LOAD DATASET
        # ==============================================

        df = pd.read_csv(
            file_path
        )

        # ==============================================
        # DETECT TARGET
        # ==============================================

        detector = TargetDetector(
            df
        )

        target_result = detector.analyze()

        target = (
            target_result[
                "selected_target"
            ]
        )

        if target is None:

            raise HTTPException(
                status_code=400,
                detail="No suitable target column detected."
            )

        problem_type = (
            target_result[
                "problem"
            ][
                "problem_type"
            ]
        )

        # ==============================================
        # ERROR ANALYSIS
        # ==============================================

        analyzer = ErrorAnalyzer(
            model_path=model_path
        )

        error_result = analyzer.analyze(
            dataframe=df,
            target_column=target,
            problem_type=problem_type
        )

        # ==============================================
        # RETURN RESULT
        # ==============================================

        return {

            "success": True,

            "dataset":
                filename,

            "target":
                target,

            "problem_type":
                problem_type,

            "error_analysis":
                error_result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/what-if")
async def what_if_analysis(request: WhatIfRequest):
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    model_path = "models/best_model_pipeline.pkl"

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Upload the dataset first."
        )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Train the model first."
        )

    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # Detect target automatically
        detector = TargetDetector(df)
        target_result = detector.analyze()

        target = target_result["selected_target"]

        if target is None:
            raise HTTPException(
                status_code=400,
                detail="No suitable target column detected."
            )

        # Detect columns removed during preprocessing
        planner = AutomaticPreprocessor(df, target)
        plan = planner.generate_plan()

        removed_columns = (
            plan["removed_columns"]["id_like"]
            + plan["removed_columns"]["constant"]
        )

        # Validate selected feature
        if request.feature not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{request.feature}' not found in dataset."
            )

        if request.feature == target:
            raise HTTPException(
                status_code=400,
                detail="Target column cannot be modified in What-If analysis."
            )

        if request.feature in removed_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{request.feature}' is removed during preprocessing and cannot be modified."
            )

        # Keep only features actually used by the model
        usable_features = (
            plan["features"]["numerical"]
            + plan["features"]["categorical"]
        )

        # Validate that required model features are provided
        missing_features = [
            column
            for column in usable_features
            if column not in request.row
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing input features: {missing_features}"
            )

        # Create original input using only model features
        original_input = {
            column: request.row[column]
            for column in usable_features
        }

        # Create modified input
        modified_input = original_input.copy()
        modified_input[request.feature] = request.new_value

        # Run What-If analysis
        engine = WhatIfEngine(model_path=model_path)

        result = engine.compare(
            original_data=original_input,
            modified_data=modified_input
        )

        return {
            "success": True,
            "dataset": request.filename,
            "target": target,
            "feature_changed": request.feature,
            "new_value": request.new_value,
            "what_if_analysis": result
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

@app.post("/report")
async def generate_report(request: ReportRequest):

    file_path = os.path.join(
        UPLOAD_DIR,
        request.filename
    )

    model_path = "models/best_model_pipeline.pkl"

    # Check dataset
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset not found. Upload the dataset first."
        )

    # Check trained model
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Train the model first."
        )

    try:

        # ==============================================
        # LOAD DATASET
        # ==============================================

        df = pd.read_csv(file_path)

        # ==============================================
        # DETECT TARGET
        # ==============================================

        detector = TargetDetector(df)

        target_result = detector.analyze()

        target = target_result["selected_target"]

        if target is None:
            raise HTTPException(
                status_code=400,
                detail="No suitable target column detected."
            )

        problem_type = target_result["problem"]["problem_type"]


        # ==============================================
        # AUTOMATIC PREPROCESSING PLAN
        # ==============================================

        planner = AutomaticPreprocessor(
            df,
            target
        )

        plan = planner.generate_plan()

        numerical_features = (
            plan["features"]["numerical"]
        )

        categorical_features = (
            plan["features"]["categorical"]
        )

        removed_columns = (
            plan["removed_columns"]["id_like"]
            +
            plan["removed_columns"]["constant"]
        )

        # ==============================================
        # BUILD PREPROCESSING PIPELINE
        # ==============================================

        processor = PreprocessingPipeline(
            df,
            target,
            numerical_features,
            categorical_features,
            removed_columns
        )

        preprocessing_pipeline = (
            processor.build()
        )

        X, y = (
            processor.prepare_data()
        )

        # ==============================================
        # MODEL TRAINING
        # ==============================================

        trainer = ModelTrainer(
            preprocessing_pipeline,
            X,
            y,
            problem_type
        )

        training_result = (
            trainer.train()
        )

        # ==============================================
        # SAVE BEST MODEL
        # ==============================================

        model_path = (
            trainer.save_model()
)
        # ==============================================
        # ERROR ANALYSIS
        # ==============================================

        error_analyzer = ErrorAnalyzer(
            model_path=model_path
        )

        error_result = error_analyzer.analyze(
            dataframe=df,
            target_column=target,
            problem_type=problem_type
        )

        # ==============================================
        # FEATURE IMPORTANCE
        # ==============================================

        importance_engine = FeatureImportanceEngine(model_path=model_path)
        importance_result = importance_engine.calculate_importance()

        # SHAP explanation
        shap_engine = ExplainabilityEngine(
            model_path=model_path,
            background_path=file_path
        )

        # Use the first row of the dataset as a representative example
        sample_input = df.drop(columns=[target], errors="ignore").iloc[0].to_dict()

        shap_result = shap_engine.explain_prediction(
            input_data=sample_input
        )

        report_engine = ModelInvestigationReport(
            training_result=training_result,
            error_result=error_result,
            importance_result=importance_result,
            shap_result=shap_result,
            target=target,
            problem_type=problem_type
        )

        # What-If Analysis
        what_if_engine = WhatIfEngine(model_path=model_path)

        original_input = df.drop(
            columns=[target],
            errors="ignore"
        ).iloc[0].to_dict()

        modified_input = original_input.copy()

        # Automatically modify one numerical feature
        numerical_columns = plan["features"]["numerical"]

        if numerical_columns:
            feature_to_modify = numerical_columns[0]
            original_value = float(modified_input[feature_to_modify])

            # Increase the feature by 20%
            modified_input[feature_to_modify] = original_value * 1.20

            what_if_result = what_if_engine.compare(
                original_data=original_input,
                modified_data=modified_input
            )
        else:
            what_if_result = None


        report = report_engine.generate_report()

        return {
            "success": True,
            "dataset": request.filename,
            "report": report
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )