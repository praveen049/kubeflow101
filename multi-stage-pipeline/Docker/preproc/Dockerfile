FROM python:3.7
COPY . /preproc
WORKDIR /preproc
RUN export http_proxy="http://10.144.1.10:8080" && export https_proxy="$http_proxy" && pip install -r requirements.txt
ENTRYPOINT [ "python", "preprocessor.py" ]