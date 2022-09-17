FROM python:3.10.5
WORKDIR /src/app
COPY src .
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python","main.py"]
