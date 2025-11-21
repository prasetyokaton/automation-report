# Gunakan image hasil build base (yang sudah punya Streamlit & lainnya)
FROM streamlit-base:latest

# Hindari buffering agar log langsung muncul
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
#COPY requirements.txt .
#RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi project ke container
COPY . .

# Expose port Streamlit
EXPOSE 8501

# Jalankan aplikasi
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
