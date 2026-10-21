# Machine Learning Zoomcamp 2026 — Homework 5

## Deploying Machine Learning Models with FastAPI and Docker

This project is my solution to Homework 5 of the **Machine Learning Zoomcamp 2026**, organized by DataTalksClub.

In this homework, I deployed a pre-trained lead conversion prediction model using FastAPI and Docker. I verified the model artifact, reproduced the reference environment, tested local predictions, and validated the containerized API.

## 1. Project Objectives

My main objectives were to:

- Set up a reproducible Python environment using `uv`.
- Verify the integrity of the provided machine learning model.
- Load the model and generate conversion probability predictions.
- Expose the model through a FastAPI REST service.
- Containerize the application using Docker.
- Validate prediction consistency between local and containerized environments.

## 2. Technologies Used

| Technology          | Version / Purpose            |
| ------------------- | ---------------------------- |
| Python              | 3.11.15                      |
| Scikit-Learn        | 1.7.2                        |
| FastAPI             | 0.119.1                      |
| Uvicorn             | 0.38.0                       |
| uv                  | 0.10.8 locally               |
| Docker              | 29.1.3                       |
| Docker Python Image | python:3.11.15-slim-bookworm |
| Docker uv Image     | 0.10.11                      |
| Git & GitHub        | Version control              |

## 3. Machine Learning Model

The provided reference model uses a Scikit-Learn pipeline consisting of:

- `DictVectorizer` for feature transformation.
- `LogisticRegression` with `solver="liblinear"` and `C=1.0` for binary classification.

The model predicts whether a potential customer will convert.

### Input Features

| Feature                  | Type        |
| ------------------------ | ----------- |
| lead_source              | Categorical |
| industry                 | Categorical |
| employment_status        | Categorical |
| location                 | Categorical |
| number_of_courses_viewed | Integer     |
| annual_income            | Numerical   |
| interaction_count        | Integer     |
| lead_score               | Numerical   |

**Target variable:** `converted`

The homework uses the frozen 2026 reference model rather than retraining the model.

## 4. Model Integrity Verification

I verified the SHA-256 checksum of the provided `pipeline.bin` artifact before loading it.

```powershell
Get-FileHash .\pipeline.bin -Algorithm SHA256
```

Verified checksum:

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```

**Result:** PASS — The model checksum matches the official reference artifact.

## 5. Environment Setup

I installed the required Python version:

```powershell
uv python install 3.11.15
```

Then I installed the dependencies using the existing lockfile:

```powershell
uv sync --locked
```

I verified the environment:

```powershell
uv run python --version
uv run python -c "import sklearn; print(sklearn.__version__)"
```

Verified versions:

```text
Python 3.11.15
Scikit-Learn 1.7.2
```

I did not regenerate the reference lockfile.

## 6. Model Prediction — Question 3

I created `q3_predict.py` to load the verified model using `pickle` and calculate the probability of conversion for the provided lead.

```powershell
uv run python q3_predict.py
```

Output:

```text
Conversion probability: 0.532994
Question 3 answer: 0.533
```

**Verified answer: 0.533**

## 7. FastAPI Deployment — Question 4

I started the prediction API using Uvicorn:

```powershell
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

The application provides two endpoints:

| Endpoint   | Method | Description                                  |
| ---------- | ------ | -------------------------------------------- |
| `/health`  | GET    | Checks service health and artifact integrity |
| `/predict` | POST   | Returns the conversion prediction            |

I submitted the Question 4 lead data to the prediction endpoint.

API response:

```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```

**Verified answer: 0.770**

## 8. Docker Deployment — Questions 5 and 6

The official Dockerfile uses the following Python base image:

```dockerfile
FROM python:3.11.15-slim-bookworm
```

The reference Dockerfile additionally pins the image using its SHA-256 digest.

I built the Docker image:

```powershell
docker build -t zoomcamp-model:2026-hw5 .
```

**Build result:** SUCCESS — 13/13 steps completed.

I started the container:

```powershell
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

I tested the containerized prediction service:

```powershell
uv run python q6_test.py
```

Output:

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

**Verified answer: 0.770**

The Docker prediction matches the local FastAPI prediction.

## 9. API Health Verification

I checked the containerized API using:

```powershell
Invoke-RestMethod -Uri "http://localhost:9696/health" -Method Get
```

Results:

- API status: `ok`
- Model checksum: verified
- Feature defaults checksum: returned successfully

**Result: PASS**

## 10. Homework Answers

| Question                  | Answer                           |
| ------------------------- | -------------------------------- |
| Q1 — uv version           | 0.10.8 (local environment check) |
| Q2 — Scikit-Learn version | 1.7.2                            |
| Q3 — Model prediction     | 0.533                            |
| Q4 — FastAPI prediction   | 0.770                            |
| Q5 — Docker base image    | python:3.11.15-slim-bookworm     |
| Q6 — Docker prediction    | 0.770                            |

Question 1 is an environment check rather than a graded model answer.

## 11. Validation Summary

| Test                                    | Result |
| --------------------------------------- | ------ |
| Python environment verification         | PASS   |
| Locked dependency installation          | PASS   |
| Model SHA-256 verification              | PASS   |
| Reference smoke test                    | PASS   |
| Question 3 prediction                   | PASS   |
| FastAPI health check                    | PASS   |
| FastAPI prediction                      | PASS   |
| Docker image build                      | PASS   |
| Docker health check                     | PASS   |
| Containerized prediction                | PASS   |
| Local vs. Docker prediction consistency | PASS   |

## 12. Key Learnings

Through this homework, I gained practical experience in deploying machine learning models beyond the training environment.

I learned how to:

- Manage reproducible Python environments with `uv`.
- Verify machine learning artifacts using cryptographic checksums.
- Load serialized Scikit-Learn pipelines and generate predictions.
- Serve machine learning models through REST APIs.
- Package prediction services into Docker containers.
- Validate consistent predictions across deployment environments.

This assignment strengthened my understanding of reproducible machine learning inference and practical model deployment.

## 13. References

- [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [DataTalksClub](https://datatalks.club/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [uv Documentation](https://docs.astral.sh/uv/)

---

**Course:** Machine Learning Zoomcamp 2026  
**Module:** 5 — Deploying Machine Learning Models  
**Project:** Lead Conversion Prediction API  
**Status:** Completed and locally validated
