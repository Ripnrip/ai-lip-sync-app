# Makefile for AI Lip Sync App

.PHONY: setup run clean

setup:
	bash setup_venv.sh

run:
	source .venv/bin/activate && streamlit run app.py

clean:
	rm -rf .venv 