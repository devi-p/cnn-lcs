# ============================================================
# Multi-stage Dockerfile for Hugging Face Docker Space
# Serves frontend + backend behind Nginx on port 7860
# ============================================================

# ---------- Stage 1: Build Next.js frontend ----------
FROM node:20-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --prefer-offline

COPY frontend/ ./
ENV NEXT_PUBLIC_API_BASE_URL=""
ENV BACKEND_URL="http://127.0.0.1:8000"
RUN npm run build

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

# Install Node.js 20 and Nginx
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl gnupg nginx libsndfile1 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python dependencies (cached layer) ----------
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ---------- Copy backend source ----------
COPY backend/ backend/
COPY src/ src/

# ---------- Copy model artifacts ----------
COPY outputs/checkpoints/ outputs/checkpoints/
COPY outputs/lcs/ outputs/lcs/
COPY outputs/interpretability/ outputs/interpretability/

# ---------- Copy built frontend ----------
COPY --from=frontend-build /build/frontend/.next frontend/.next
COPY --from=frontend-build /build/frontend/node_modules frontend/node_modules
COPY --from=frontend-build /build/frontend/package.json frontend/package.json
COPY --from=frontend-build /build/frontend/public frontend/public
COPY --from=frontend-build /build/frontend/next.config.ts frontend/next.config.ts

# ---------- Copy Nginx config and start script ----------
COPY nginx.conf /app/nginx.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---------- Create temp dirs for Nginx (non-root) ----------
RUN mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy \
    /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi

# ---------- Environment ----------
ENV PYTHONUNBUFFERED=1
ENV BACKEND_URL="http://127.0.0.1:8000"
ENV NEXT_PUBLIC_API_BASE_URL=""
ENV API_ALLOW_ORIGINS="*"
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

EXPOSE 7860

CMD ["/app/start.sh"]
