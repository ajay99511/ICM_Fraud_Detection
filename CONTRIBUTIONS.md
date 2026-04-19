# Project Contributions: IEEE-CIS Fraud Detection

This document summarizes the analysis, testing, and technical improvements contributed to the IEEE-CIS Fraud Detection project.

## 1. Project Analysis
The project is a machine learning-based fraud detection system consisting of:
- **FastAPI Application (`app.py`)**: Exposes a REST API for real-time fraud prediction.
- **Machine Learning Pipeline (`src/pipeline/`)**: Handles data preprocessing (time engineering, encoding) and model inference.
- **Modular Components (`src/components/`)**: Includes dedicated scripts for data ingestion, transformation, and model training.
- **Technologies Used**: Python 3.13, FastAPI, XGBoost, LightGBM, Scikit-Learn, Pandas, NumPy, and Pydantic.

## 2. Work Accomplished

### Automated Testing Suite
- Created a comprehensive test suite in the `tests/` directory using the `unittest` framework.
- **Unit Tests (`tests/test_pipeline.py`)**: Verified the `PredictionPipeline` preprocessing logic and prediction output consistency.
- **Integration Tests (`tests/test_app.py`)**: Validated the FastAPI endpoints (`/` and `/predict`) using `fastapi.testclient.TestClient`.
- Ensured tests are environment-aware (skipping tests if models or data samples are missing).

### Bug Discovery & Diagnostics
- **LabelEncoder Bug**: Identified a critical issue in `src/pipeline/prediction_pipeline.py` where `LabelEncoder.fit_transform()` was being used on single inference rows. This results in arbitrary encoding that doesn't match the training data.
- **Pydantic Deprecation**: Noticed the use of the deprecated `.dict()` method in the Pydantic model in `app.py`.
- **Performance Bottlenecks**: Identified `DataFrame` fragmentation warnings in the prediction pipeline caused by iterative column assignments.

### Environment Management
- Diagnosed and resolved Python environment conflicts (Python 3.13 vs 3.14) to ensure the test suite executed with the correct dependencies.

## 3. Skills & Contributions
- **System Architecture Analysis**: Mapping complex ML workflows to identify potential points of failure.
- **Quality Assurance**: Implementing automated testing for ML-driven APIs.
- **Machine Learning Engineering**: Analyzing preprocessing consistency between training and inference phases.
- **Backend Development**: Validating FastAPI schemas and endpoint behavior.
- **Technical Documentation**: Creating clear, actionable reports on system health and improvements.

## 4. Technical Recommendations
1. **Fix Categorical Encoding**: Replace the on-the-fly `LabelEncoder.fit_transform()` in the prediction pipeline with pre-fitted encoders (e.g., using `joblib` to save encoders during the transformation step).
2. **Modernize Pydantic**: Update `transaction.dict()` to `transaction.model_dump()` in `app.py`.
3. **Optimize DataFrame Operations**: Use `pd.concat` instead of iterative assignments to resolve fragmentation warnings.
4. **CI/CD Integration**: Add the `python -m unittest discover tests` command to the CI pipeline to ensure regression-free updates.
