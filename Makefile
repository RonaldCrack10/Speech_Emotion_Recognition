
PYTHON = python3

all: run

run:
	$(PYTHON) main.py

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) pylint

# test:
# 	$(PYTHON) -m pytest


