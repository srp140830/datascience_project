# End-to-End Data Science Project Workflow 🚀

Built with best practices in mind, the pipeline integrates tools such as MLflow for experiment tracking and DagsHub for version control and collaboration, making it suitable for real-world, scalable applications.
This repository provides a complete, modular, and production-ready **Machine Learning Pipeline** covering everything from data ingestion to model evaluation using tools like **MLflow** and **DagsHub**.

## Project Structure

The pipeline includes the following key components:

1. **Data Ingestion**
2. **Data Validation**
3. **Data Transformation**
4. **Feature Engineering & Preprocessing**
5. **Model Training**
6. **Model Evaluation**

Each stage is modularized and can be updated independently.

---

## Tools & Technologies

- **MLflow** – for experiment tracking and model registry
- **DagsHub** – for version control of data, models, experiments
- **YAML configs** – for structured configuration
- **Scikit-learn / Pandas / NumPy** – for ML and preprocessing
- **Python OOP** – for clean and maintainable codebase

---

## Workflow for Updating the Pipeline

Follow the below sequence to implement or update any part of the ML pipeline:

### 1️. Update `config.yaml`
- Define global settings and paths used across the project.

### 2. Update `schema.yaml`
- Specify the expected structure and data types for your dataset (used in data validation).

### 3️. Update `params.yaml`
- Define all model parameters for consistency and tuning.

### 4️. Update the **Entity Classes**
- These define the data models (input/output contracts) used by each component.

### 5️. Update the **Configuration Manager** (`src/config/configuration.py`)
- Reads and parses the YAML files and provides structured configuration to each component.

### 6️. Update the **Pipeline Components**
- These include:
  - `data_ingestion.py`
  - `data_validation.py`
  - `data_transformation.py`
  - `model_trainer.py`
  - `model_evaluation.py`

### 7️. Update the **Pipeline Orchestration Code**
- Located in `src/pipeline/`, update the flow logic to include new/modified components.

### 8️. Update `main.py`
- The entry point of the pipeline where all stages are triggered in sequence.

---

## Running the Pipeline

```bash
python main.py
