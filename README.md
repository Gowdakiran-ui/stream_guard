---

# 🚨 StreamGuard – Real-Time Fraud Detection with MLOps

## 📌 Overview

**StreamGuard** is an end-to-end **real-time fraud detection system** built using modern **MLOps practices**.
It ingests streaming data through **Kafka**, processes and stores it in **PostgreSQL**, applies **EDA & Feature Engineering**, trains an **ML model**, and deploys the model inside a **containerized, CI/CD pipeline** using **Jenkins, Docker, and Kubernetes**.

The project simulates **real-world financial fraud detection** pipelines used in banks and fintech.

---

## ⚡ Features

* **📥 Real-time Data Ingestion** via **Apache Kafka**
* **🗄️ Reliable Storage** with **PostgreSQL**
* **🔍 Data Analysis** – EDA & Feature Engineering
* **🤖 ML Model Training** (Fraud Classification)
* **⚙️ CI/CD Pipeline** with **Jenkins**
* **🐳 Containerization** using **Docker**
* **☸️ Orchestration** using **Kubernetes**
* **📊 Monitoring** with **Prometheus** (future scope)
* **🌐 Web UI** to visualize predictions & fraud alerts

---

## 🏗️ Architecture

```mermaid
flowchart LR
  A[Kafka Producer] --> B[Kafka Broker]
  B --> C[Kafka Consumer]
  C --> D[PostgreSQL DB]
  D --> E[ML Pipeline (EDA + Feature Engg + Training)]
  E --> F[Model Registry / MLflow]
  F --> G[Dockerized Model Service]
  G --> H[Kubernetes Cluster]
  H --> I[Flask/HTML UI]
  G --> J[Jenkins CI/CD]
```

---

## 🛠️ Tech Stack

| Component               | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| **Kafka**               | Stream ingestion of transaction logs             |
| **PostgreSQL**          | Data storage for historical & streaming data     |
| **EDA & Feature Engg.** | Data preprocessing for fraud detection           |
| **ML Model**            | Fraud classification (binary: fraud / not fraud) |
| **MLflow**              | Model tracking & versioning                      |
| **Docker**              | Containerization of services                     |
| **Kubernetes**          | Deployment & orchestration                       |
| **Jenkins**             | CI/CD automation                                 |
| **Prometheus**          | Monitoring & alerting                            |
| **Flask / HTML**        | Web UI for fraud alerts & pipeline visualization |

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Gowdakiran-ui/streamguard-ml.git
cd streamguard-ml
```

### 2️⃣ Run Kafka + Zookeeper

```bash
docker-compose -f kafka/docker-compose.yml up -d
```

### 3️⃣ Load Dataset into PostgreSQL

```bash
python data.py   # Upload IEEE Fraud Dataset into PostgreSQL
```

### 4️⃣ Train ML Model

Run training in **Google Colab** or locally:

```python
python train.py
```

### 5️⃣ Deploy Model with Docker + Kubernetes

```bash
docker build -t streamguard-ml .
kubectl apply -f k8s-deployment.yaml
```

### 6️⃣ CI/CD with Jenkins

* Add pipeline in Jenkinsfile
* Auto build & deploy on commit

---

## 📊 Example Fraud Log (UI Output)

```
[2025-09-02 10:21:15]  💳 TransactionID=293881 Fraud=❌  
[2025-09-02 10:21:16]  💳 TransactionID=293882 Fraud=⚠️ SCAM FLAGGED  
[2025-09-02 10:21:17]  💳 TransactionID=293883 Fraud=❌  
```

---

## 📂 Repository Structure

```
streamguard-ml/
│── kafka/
│   ├── producer.js
│   ├── consumer.js
│   └── docker-compose.yml
│── data.py          # Load dataset → PostgreSQL
│── train.py         # Model training & MLflow logging
│── app.py           # Flask backend
│── templates/       # HTML UI
│── Dockerfile
│── Jenkinsfile
│── k8s-deployment.yaml
│── README.md
```

---

## 📌 Future Improvements

* ✅ Model monitoring with **Prometheus & Grafana**
* ✅ Advanced feature engineering with **transaction graphs**
* ✅ Deploy with **FastAPI** for better scalability

---

## 👨‍💻 Author

**Kiran Gowda A**

* 🌐 [Portfolio](https://kiran-gowda-persona.vercel.app)
* 💼 [LinkedIn](https://www.linkedin.com/in/kiran-gowda-80786824a)
* 💻 [GitHub](https://github.com/Gowdakiran-ui)

---



