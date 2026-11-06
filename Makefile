.PHONY: setup data eda majority tfidf sbert compare figures errors readme site demo test all

setup:
	pip install -r requirements.txt

data:
	python -m src.data_loading && python -m src.preprocess

eda:
	python -m src.eda

majority:
	python -m src.train_majority

tfidf:
	python -m src.train_tfidf --mode full
	python -m src.train_tfidf --mode claim_only
	python -m src.train_tfidf --mode evidence_only

sbert:
	python -m src.train_sbert --mode full
	python -m src.train_sbert --mode claim_only
	python -m src.train_sbert --mode evidence_only

compare:
	python -m src.compare

figures:
	python -m src.figures

errors:
	python -m src.error_analysis

readme:
	python -m src.readme_results

site:
	python site/build_site.py

demo:
	streamlit run app/streamlit_app.py

test:
	pytest -q

all: setup data eda majority tfidf sbert compare figures errors readme site test
