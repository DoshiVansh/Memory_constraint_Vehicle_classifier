pipeline {
    agent { label 'kubernetes-deployer' }

    options {
        disableConcurrentBuilds()
        timestamps()
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    parameters {
        text(name: 'RELEASE_METADATA_JSON', defaultValue: '', description: 'Paste the complete release_metadata.json attached to the GHCR GitHub Release (optional if RELEASE_TAG is provided)')
        string(name: 'RELEASE_TAG', defaultValue: '', description: 'GitHub Release tag (e.g. v1.0.0 or latest) to fetch metadata automatically')
    }

    environment {
        KUBECONFIG = credentials('vehicle-classifier-kubeconfig')
        K8S_NAMESPACE = 'vehicle-classifier-staging'
        DEPLOYMENT_NAME = 'vehicle-classifier-api'
        SERVICE_NAME = 'vehicle-classifier-api'
        PREVIOUS_REVISION = ''
        PREVIOUS_MODEL_VERSION = ''
        PREVIOUS_GIT_SHA = ''
        PREVIOUS_MODEL_SHA256 = ''
        DEPLOYMENT_EXISTED = 'false'
        DEPLOYMENT_ATTEMPTED = 'false'
    }

    stages {
        stage('Validate release identity') {
            steps {
                script {
                    if (params.RELEASE_METADATA_JSON?.trim()) {
                        writeFile file: 'release_metadata.json', text: params.RELEASE_METADATA_JSON
                    } else {
                        def tag = params.RELEASE_TAG?.trim() ?: 'latest'
                        sh """#!/bin/sh
                            set -eu
                            echo "Fetching release metadata automatically for tag '${tag}'..."
                            if [ "${tag}" = "latest" ]; then
                              RELEASE_URL="https://api.github.com/repos/\${GITHUB_REPOSITORY:-bhavsarnaitik4/Vehicle_classifier_bucket}/releases/latest"
                            else
                              RELEASE_URL="https://api.github.com/repos/\${GITHUB_REPOSITORY:-bhavsarnaitik4/Vehicle_classifier_bucket}/releases/tags/${tag}"
                            fi
                            if curl -fsSL -H "Accept: application/vnd.github.v3+json" "\$RELEASE_URL" 2>/dev/null | grep -o 'https://[^"]*release_metadata.json' | head -n 1 | xargs -r curl -fsSL -o release_metadata.json 2>/dev/null; then
                              echo "Successfully downloaded release_metadata.json from GitHub Release."
                            elif [ -f reports/release_metadata.json ]; then
                              echo "Using local reports/release_metadata.json as fallback."
                              cp reports/release_metadata.json release_metadata.json
                            else
                              echo "Error: No RELEASE_METADATA_JSON provided and could not fetch release_metadata.json for tag ${tag}." >&2
                              exit 1
                            fi
                        """
                    }
                }
                sh 'python3 scripts/validate_release_metadata.py release_metadata.json --output release-identity.env'
                script {
                    def identity = readFile('release-identity.env').readLines().collectEntries { line ->
                        def parts = line.split('=', 2)
                        [(parts[0]): parts.length > 1 ? parts[1] : '']
                    }
                    identity.each { key, value -> env[key] = value }
                }
            }
        }

        stage('Preflight cluster') {
            steps {
                sh '''#!/bin/sh
                    set -eu
                    kubectl version --client
                    kubectl cluster-info
                    kubectl apply -f k8s/namespace.yaml
                    kubectl get namespace "$K8S_NAMESPACE"
                    if kubectl get deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" >/dev/null 2>&1; then
                      DEPLOYMENT_EXISTED=true
                      PREVIOUS_REVISION=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" -o jsonpath='{.metadata.annotations.deployment\\.kubernetes\\.io/revision}')
                      test -n "$PREVIOUS_REVISION"
                      PREVIOUS_MODEL_VERSION=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" -o jsonpath='{.spec.template.metadata.labels.model-version}')
                      PREVIOUS_GIT_SHA=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" -o jsonpath='{.spec.template.metadata.labels.release-git-sha}')
                      PREVIOUS_MODEL_SHA256=$(kubectl get configmap vehicle-classifier-config -n "$K8S_NAMESPACE" -o jsonpath='{.data.MODEL_SHA256}')
                      echo "Found existing deployment revision $PREVIOUS_REVISION."
                    else
                      echo 'No previous staging deployment exists; a failed first rollout will be removed.'
                    fi
                    echo "DEPLOYMENT_EXISTED=$DEPLOYMENT_EXISTED" >> "$WORKSPACE/deploy-state.env"
                    echo "PREVIOUS_REVISION=$PREVIOUS_REVISION" >> "$WORKSPACE/deploy-state.env"
                    echo "PREVIOUS_MODEL_VERSION=$PREVIOUS_MODEL_VERSION" >> "$WORKSPACE/deploy-state.env"
                    echo "PREVIOUS_GIT_SHA=$PREVIOUS_GIT_SHA" >> "$WORKSPACE/deploy-state.env"
                    echo "PREVIOUS_MODEL_SHA256=$PREVIOUS_MODEL_SHA256" >> "$WORKSPACE/deploy-state.env"
                '''
                script {
                    def state = readFile('deploy-state.env').readLines().collectEntries { line ->
                        def parts = line.split('=', 2)
                        [(parts[0]): parts.length > 1 ? parts[1] : '']
                    }
                    env.DEPLOYMENT_EXISTED = state.DEPLOYMENT_EXISTED
                    env.PREVIOUS_REVISION = state.PREVIOUS_REVISION
                    env.PREVIOUS_MODEL_VERSION = state.PREVIOUS_MODEL_VERSION
                    env.PREVIOUS_GIT_SHA = state.PREVIOUS_GIT_SHA
                    env.PREVIOUS_MODEL_SHA256 = state.PREVIOUS_MODEL_SHA256
                }
            }
        }

        stage('Deploy staging digest') {
            steps {
                script { env.DEPLOYMENT_ATTEMPTED = 'true' }
                sh '''#!/bin/sh
                    set -eu
                    export IMAGE_REF MODEL_VERSION MODEL_SHA256 RELEASE_GIT_SHA
                    envsubst '${MODEL_VERSION} ${MODEL_SHA256} ${RELEASE_GIT_SHA}' < k8s/configmap.yaml.tpl | kubectl apply -f -
                    envsubst '${IMAGE_REF} ${MODEL_VERSION} ${RELEASE_GIT_SHA}' < k8s/deployment.yaml.tpl | kubectl apply -f -
                    kubectl apply -f k8s/service.yaml
                    if [ -f k8s/hpa.yaml ]; then
                      kubectl apply -f k8s/hpa.yaml
                    fi
                    kubectl rollout status "deployment/$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" --timeout=4m
                    kubectl get deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" -o wide
                '''
            }
        }

        stage('Readiness and prediction smoke') {
            steps {
                sh '''#!/bin/sh
                    set -eu
                    PORT_FORWARD_LOG=$(mktemp)
                    kubectl port-forward -n "$K8S_NAMESPACE" "service/$SERVICE_NAME" 18000:8000 >"$PORT_FORWARD_LOG" 2>&1 &
                    PORT_FORWARD_PID=$!
                    cleanup() {
                      kill "$PORT_FORWARD_PID" 2>/dev/null || true
                      rm -f "$PORT_FORWARD_LOG"
                    }
                    trap cleanup EXIT INT TERM
                    python3 scripts/k8s_smoke_test.py --base-url http://127.0.0.1:18000 --timeout-seconds 120
                '''
            }
        }
    }

    post {
        success {
            echo "Staging release succeeded: ${params.IMAGE_REF} (model ${params.MODEL_VERSION}, commit ${params.RELEASE_GIT_SHA})."
        }
        unsuccessful {
            script {
                if (env.DEPLOYMENT_ATTEMPTED == 'true') {
                    if (env.DEPLOYMENT_EXISTED == 'true' && env.PREVIOUS_REVISION?.trim()) {
                        echo "Deployment failed; restoring known-good revision ${env.PREVIOUS_REVISION}."
                        int rollbackStatus = sh(
                            script: '''#!/bin/sh
                                set +e
                                export MODEL_VERSION="$PREVIOUS_MODEL_VERSION" MODEL_SHA256="$PREVIOUS_MODEL_SHA256" RELEASE_GIT_SHA="$PREVIOUS_GIT_SHA"
                                envsubst '${MODEL_VERSION} ${MODEL_SHA256} ${RELEASE_GIT_SHA}' < k8s/configmap.yaml.tpl | kubectl apply -f -
                                kubectl rollout undo "deployment/$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" --to-revision="$PREVIOUS_REVISION"
                                undo_status=$?
                                if [ "$undo_status" -ne 0 ]; then exit "$undo_status"; fi
                                kubectl rollout status "deployment/$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" --timeout=4m
                                PORT_FORWARD_LOG=$(mktemp)
                                kubectl port-forward -n "$K8S_NAMESPACE" "service/$SERVICE_NAME" 18000:8000 >"$PORT_FORWARD_LOG" 2>&1 &
                                PORT_FORWARD_PID=$!
                                trap 'kill "$PORT_FORWARD_PID" 2>/dev/null || true; rm -f "$PORT_FORWARD_LOG"' EXIT INT TERM
                                python3 scripts/k8s_smoke_test.py --base-url http://127.0.0.1:18000 --timeout-seconds 120
                            ''',
                            returnStatus: true
                        )
                        if (rollbackStatus != 0) {
                            error("Automatic rollback failed (kubectl exit ${rollbackStatus}); staging needs operator recovery.")
                        }
                        echo 'Previous deployment revision is restored and available.'
                    } else {
                        echo 'First deployment failed; removing the unverified Deployment.'
                        sh 'kubectl delete deployment "$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" --ignore-not-found=true'
                    }
                }
            }
        }
        always {
            sh 'rm -f deploy-state.env release_metadata.json release-identity.env'
            echo 'See console output for the deployed image digest and rollout/smoke result.'
        }
    }
}
