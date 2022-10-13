FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["predict_hw5.py", "model1.bin", "dv.bin", "./"]

EXPOSE 7777

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:7777", "predict_hw5:app"]