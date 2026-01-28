pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                git 'https://github.com/anjilinux/project-mlops-mlflow-jenkins-World-Population-Prediction-Linear-Regression.git'
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Test Model') {
            steps {
                sh '''
                . venv/bin/activate
                pytest test_model.py
                '''
            }
        }

        stage('Predict') {
            steps {
                sh '''
                . venv/bin/activate
                python predict.py
                '''
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline completed successfully"
        }
        failure {
            echo "❌ Pipeline failed"
        }
    }
}
