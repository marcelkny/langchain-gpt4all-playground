#!/bin/sh
export FLASK_APP=./cashman/server.py
pipenv run flask --debug run -h 0.0.0.0
