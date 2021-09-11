PYTHON        := $(shell which python)

SOURCE       ?= .
RSTC_FILES   := $(shell find ${SOURCE} -name *.rstc)
RST_RESULTS  := $(addsuffix .auto.rst, $(basename ${RSTC_FILES}))

APIDOC_GEN_PY := $(shell readlink -f ${SOURCE}/apidoc_gen.py)

%.auto.rst: %.rstc ${APIDOC_GEN_PY}
	cd "$(shell dirname $(shell readlink -f $<))" && \
		PYTHONPATH="$(shell dirname $(shell readlink -f $<)):${PYTHONPATH}" \
		cat "$(shell readlink -f $<)" | $(PYTHON) "${APIDOC_GEN_PY}" > "$(shell readlink -f $@)"

build: ${RST_RESULTS}

all: build

clean:
	rm -rf \
		$(shell find ${SOURCE} -name *.auto.rst)
