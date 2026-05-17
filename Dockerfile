# DOCKER CONTAINER SET UP FOR THE DIABETES PREDICTION

# ============================
# 1. BASE IMAGE
# ============================
FROM python:3.10-slim

# ============================
# 2. SET THE WORKING DIRECTORY
# ============================
WORKDIR /app

# ===========================
# 3. COPY PROJECT FILES
# ===========================
COPY . /app

# ===========================
# 4. INSTALL DEPENDENCIES
# ===========================
RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# 5. EXPOSE PORT FOR FASTAPI
# ===========================
EXPOSE 8000

# ==========================
# 6. RUN FASTAPI APP
# ==========================
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
