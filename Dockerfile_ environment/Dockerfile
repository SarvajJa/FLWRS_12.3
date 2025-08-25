FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential libfftw3-dev
WORKDIR /opt/flwrs
COPY requirements.txt /opt/flwrs/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /opt/flwrs
ENV PYTHONPATH=/opt/flwrs/src
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
24.4 Notebook checklist and tests
•Synthetic example: run
https://github.com/SarvajJa/FLWRS_12.3/blob/main/Notebooks/spectral_demo.ipynb
to reproduce figure showing initial waveform, fractional Laplacian transform, and
Luminissance effect.
•Adjoint verification: run
https://github.com/SarvajJa/FLWRS_12.3/blob/main/Notebooks/adjoint_tutorial.ipynb
and perform finite-difference check of gradients (relative error ≤10−6).
•Automated tests: pytestintests/verifying mass conservation (when λ= 0) and
convergence in α→2limit.
27