apiVersion: apps/v1
kind: Deployment
metadata:
  name: vehicle-classifier-api
  namespace: vehicle-classifier-staging
  labels:
    app.kubernetes.io/name: vehicle-classifier-api
    app.kubernetes.io/part-of: vehicle-classifier
spec:
  replicas: 1
  revisionHistoryLimit: 5
  progressDeadlineSeconds: 180
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: vehicle-classifier-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: vehicle-classifier-api
        app.kubernetes.io/part-of: vehicle-classifier
        model-version: "${MODEL_VERSION}"
        release-git-sha: "${RELEASE_GIT_SHA}"
    spec:
      automountServiceAccountToken: false
      terminationGracePeriodSeconds: 20
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: api
          image: ${IMAGE_REF}
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          envFrom:
            - configMapRef:
                name: vehicle-classifier-config
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 2
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 12
          livenessProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health/live
              port: http
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 18
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 2Gi
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: [ALL]
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 64Mi
