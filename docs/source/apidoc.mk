PYTHON := $(shell which python)

SOURCE         ?= .
PYTHON_SCRIPTS := $(shell find ${SOURCE} -name *.rst.py)
PYTHON_RESULTS := $(addsuffix .auto.rst, $(basename $(basename ${PYTHON_SCRIPTS})))

%.auto.rst: %.rst.py
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		$(PYTHON) "$(shell readlink -f $<)" > "$(shell readlink -f $@)"

build: ${PYTHON_RESULTS}

all: build

clean:
	rm -rf \
		$(shell find ${SOURCE} -name *.auto.rst)
