pipeline {
    agent any

    environment {
        PYTHON = 'python'  // or 'python3' depending on your system
    }

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main', url: 'https://github.com/Gowdakiran-ui/stream_guard.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh "${PYTHON} -m pip install --upgrade pip"
                sh "${PYTHON} -m pip install -r requirements.txt"
            }
        }

        stage('Extract Data from PostgreSQL') {
            steps {
                sh "${PYTHON} src/data/load_from_postgres.py"
            }
        }

        stage('Train Model') {
            steps {
                sh "${PYTHON} src/train.py"
            }
        }

        stage('Evaluate Model') {
            steps {
                sh "${PYTHON} src/evaluate.py"
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/outputs/**', fingerprint: true
        }
    }
}
