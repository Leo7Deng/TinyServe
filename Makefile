.PHONY: install build clean test

config:
	git config user.name "Leo Deng"
	git config user.email "leo7deng"

install:
	pip install -r requirements-dev.txt

build:
	pip install -e . --no-build-isolation

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "*.so" -delete

test:
	python tests/test_throughput.py
