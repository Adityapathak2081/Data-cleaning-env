# ─────────────────────────────────────────
# Use Python 3.11 as the base image
# (a clean Linux machine with Python ready)
# ─────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────
# Set the working directory inside container
# All our files will live here
# ─────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────
# Copy all our project files into container
# ─────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────
# Install all required Python libraries
# ─────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    openai \
    pandas \
    openenv-core

# ─────────────────────────────────────────
# Expose port 7860 (required by HF Spaces)
# ─────────────────────────────────────────
EXPOSE 7860

# ─────────────────────────────────────────
# Command to start the server
# ─────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]