PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -c "import sys; sys.path.insert(0,'src'); from airq.data import generate_and_save; generate_and_save()"

train:
	$(PY) scripts/run_pipeline.py

test:
	$(PY) -m unittest discover -s tests -v

run: train
