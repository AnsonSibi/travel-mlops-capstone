// CI/CD pipeline for the flight price prediction service
//
// Checkout -> Setup -> Lint -> Unit tests -> (Retrain) -> Model quality gate
//   -> Build images -> Smoke test container -> Push -> Deploy to Kubernetes
//
// Jenkins needs: Docker, kubectl (with a kubeconfig for the cluster) and Python 3.
// Credentials: 'dockerhub-creds' (username/password) if PUSH_IMAGE is enabled.

pipeline {
    agent any

    parameters {
        booleanParam(name: 'RETRAIN', defaultValue: false,
                     description: 'Retrain all models (logged to MLflow) before building the image')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false,
                     description: 'Push images to the registry (otherwise load them into minikube)')
        booleanParam(name: 'DEPLOY', defaultValue: true, description: 'Deploy to Kubernetes')
        string(name: 'REGISTRY', defaultValue: 'docker.io/your-dockerhub-user',
               description: 'Image registry / Docker Hub namespace')
        string(name: 'MODEL_RUN_ID', defaultValue: '',
               description: 'Set automatically when the Airflow training DAG triggers this job')
    }

    environment {
        VENV        = "${WORKSPACE}/.venv"
        API_IMAGE   = "${params.REGISTRY}/travel-ml-api"
        UI_IMAGE    = "${params.REGISTRY}/travel-ml-streamlit"
        K8S_NS      = 'travel-ml'
        MLFLOW_DISABLE_AGENT_HINT = '1'
    }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.IMAGE_TAG = "${env.BUILD_NUMBER}-${env.GIT_SHORT}"
                }
                echo "Building image tag ${env.IMAGE_TAG}"
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install -q --upgrade pip
                    pip install -q -r requirements.txt
                    pip install -q -e .
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '. "$VENV/bin/activate" && flake8 src api streamlit_app tests scripts airflow/dags'
            }
        }

        stage('Unit tests') {
            steps {
                sh '. "$VENV/bin/activate" && pytest --junitxml=reports/junit.xml'
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'reports/junit.xml' }
            }
        }

        stage('Retrain models') {
            when { expression { params.RETRAIN } }
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    python -m travel_ml.train_regression
                    python -m travel_ml.train_classifier
                    python -m travel_ml.recommender
                '''
            }
        }

        stage('Model quality gate') {
            steps {
                // never ship a model that is worse than the agreed bar
                sh '. "$VENV/bin/activate" && python scripts/check_model_quality.py --min-r2 0.95 --max-rmse 50'
            }
        }

        stage('Build Docker images') {
            steps {
                sh '''
                    docker build -f docker/Dockerfile.api -t "$API_IMAGE:$IMAGE_TAG" -t "$API_IMAGE:latest" .
                    docker build -f docker/Dockerfile.streamlit -t "$UI_IMAGE:$IMAGE_TAG" -t "$UI_IMAGE:latest" .
                '''
            }
        }

        stage('Smoke test container') {
            steps {
                sh '''
                    docker rm -f api-smoke 2>/dev/null || true
                    docker run -d --name api-smoke -p 5099:5000 "$API_IMAGE:$IMAGE_TAG"
                    for i in $(seq 1 30); do
                        curl -sf http://localhost:5099/ready && break
                        sleep 2
                    done
                    curl -sf -X POST http://localhost:5099/predict \
                         -H "Content-Type: application/json" \
                         -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops"}'
                '''
            }
            post {
                always { sh 'docker rm -f api-smoke || true' }
            }
        }

        stage('Push images') {
            when { expression { params.PUSH_IMAGE } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds',
                                                  usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
                    sh '''
                        echo "$REG_PASS" | docker login -u "$REG_USER" --password-stdin
                        docker push "$API_IMAGE:$IMAGE_TAG" && docker push "$API_IMAGE:latest"
                        docker push "$UI_IMAGE:$IMAGE_TAG" && docker push "$UI_IMAGE:latest"
                    '''
                }
            }
        }

        stage('Load images into minikube') {
            when { expression { !params.PUSH_IMAGE && params.DEPLOY } }
            steps {
                sh '''
                    minikube image load "$API_IMAGE:$IMAGE_TAG"
                    minikube image load "$UI_IMAGE:$IMAGE_TAG"
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            when { expression { params.DEPLOY } }
            steps {
                sh '''
                    kubectl apply -f k8s/namespace.yaml
                    kubectl apply -f k8s/
                    kubectl -n "$K8S_NS" set image deployment/flight-price-api api="$API_IMAGE:$IMAGE_TAG"
                    kubectl -n "$K8S_NS" set image deployment/travel-streamlit streamlit="$UI_IMAGE:$IMAGE_TAG"
                    kubectl -n "$K8S_NS" annotate deployment/flight-price-api \
                        kubernetes.io/change-cause="build $IMAGE_TAG model-run ${MODEL_RUN_ID:-n/a}" --overwrite
                    kubectl -n "$K8S_NS" rollout status deployment/flight-price-api --timeout=180s
                    kubectl -n "$K8S_NS" rollout status deployment/travel-streamlit --timeout=180s
                    kubectl -n "$K8S_NS" get pods -o wide
                '''
            }
            post {
                failure {
                    // automatic rollback to the previous ReplicaSet (previous model)
                    sh 'kubectl -n "$K8S_NS" rollout undo deployment/flight-price-api || true'
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'models/*.json, reports/*.json', allowEmptyArchive: true
        }
        success { echo "Deployed ${env.API_IMAGE}:${env.IMAGE_TAG}" }
        failure { echo 'Pipeline failed - check the stage logs above' }
    }
}
