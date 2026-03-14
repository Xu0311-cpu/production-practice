FROM python:3.9-slim
WORKDIR /app
COPY shift_calculation.py .
CMD ["python", "shift_calculation.py"]
