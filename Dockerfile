FROM python

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "job_runner.py"]
