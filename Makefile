
PYTHON = python3

all: run

run:
	$(PYTHON) app.py

install:
	$(PYTHON) -m pip install -r requirements.txt

# test:
# 	$(PYTHON) -m pytest


