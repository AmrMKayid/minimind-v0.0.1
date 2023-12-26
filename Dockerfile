FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y git openssh-client make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl ca-certificates vim


# installs python 3.10.11 through pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.10.11 && pyenv global 3.10.11
ENV PATH="/root/.pyenv/versions/3.10.11/bin:$PATH"


# adds github.com to known hosts to allow ssh dependencies download
RUN mkdir -p -m 0600  ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts


# from google docs https://cloud.google.com/sdk/docs/install
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

# allows us to read python stdout in k8 logging system in ~real time
ENV PYTHONUNBUFFERED 1


# gcloud creds
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json


# environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CLOUDSDK_PYTHON=/usr/bin/python3



RUN python3 -m pip install poetry==1.7.1

WORKDIR /minimind

COPY ./pyproject.toml ./poetry.lock ./

RUN poetry env use /root/.pyenv/versions/3.10.11/bin/python3

RUN git config --global --add safe.directory /minimind

# install packages
# RUN --mount=type=ssh poetry install --with dev -vvv
