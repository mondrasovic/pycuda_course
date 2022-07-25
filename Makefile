setup: requirements.txt
	@echo Installing requirements.
	pip install -r requirements.txt

clean:
	@echo Cleaning Python pre-compiled files.
	rm -rf __pycache__