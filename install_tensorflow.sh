#!/bin/sh
set -ex
apt-get update -qq && apt-get install -qq \
    build-essential \
    python3 \
    python-virtualenv
virtualenv -p python3 tf_env
. tf_env/bin/activate
pip install --progress-bar off --upgrade pip
pip install --progress-bar off --upgrade tensorflow
deactivate
