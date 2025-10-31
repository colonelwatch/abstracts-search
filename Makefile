DATA_DIR := abstracts-embeddings/data
INDEX_DIR := abstracts-faiss/index

-include env.mk

SHELL := bash
PYTHON := conda run -n abstracts-search --live-stream python

CFLAGS ?= -O2
INDEXFLAGS += -B $(INDEX_DIR)

.PHONY: all
INDEX_FILL_TARGETS := $(addprefix $(INDEX_DIR)/, ids.parquet index.faiss ondisk.ivfdata)
INDEX_TRAIN_TARGETS := $(addprefix $(INDEX_DIR)/, empty.faiss untuned.json)
all: $(INDEX_FILL_TARGETS) $(INDEX_DIR)/params.json

$(INDEX_FILL_TARGETS) &: $(DATA_DIR) $(INDEX_TRAIN_TARGETS)
	$(PYTHON) index.py $(INDEXFLAGS) fill $(INDEXFILLFLAGS) $(DATA_DIR)

$(INDEX_DIR)/params.json: $(INDEX_TRAIN_TARGETS) | $(DATA_DIR)
	$(PYTHON) index.py $(INDEXFLAGS) tune $(INDEXTUNEFLAGS) $(DATA_DIR)

$(INDEX_TRAIN_TARGETS) &: | $(DATA_DIR)
	$(PYTHON) index.py $(INDEXFLAGS) train $(INDEXTRAINFLAGS) $(DATA_DIR)

# TODO: better handle the already existing directory
include remote_targets.mk
EQ := =
$(DATA_DIR) abstracts-embeddings/events &: $(events)
	if [ -d $(DATA_DIR) ]; then	\
		rm -rf $(DATA_DIR);	\
	fi
	$(PYTHON) dump.py $(DUMPFLAGS) data.sqlite $(DATA_DIR)
	cp -r events abstracts-embeddings/

# in theory, wouldn't I need to handle two objects in one line here? I could probably
# update oa_jsonl to handle that?
EQ := =
events/updated_date$(EQ)% : | manifest.txt oa_jsonl data.sqlite events
	s3_base="s3://openalex/data/works"; 				\
	http_base="https://openalex.s3.amazonaws.com/data/works"; 	\
	grep "$(subst events/,,$@)" manifest.txt | 			\
		sed "s|$$s3_base|$$http_base|" | xargs -- curl -s | 	\
		gunzip | ./oa_jsonl | mbuffer -q -t -m 16G | 		\
		$(PYTHON) build.py $(BUILDFLAGS) data.sqlite
	touch $@

oa_jsonl: oa_jsonl.c
	$(CC) $(CFLAGS) -o $@ $<

data.sqlite:
	$(PYTHON) -m utils.table_utils $@

events:
	mkdir events

remote_targets.mk: manifest.txt
	tmp=$$(mktemp); 						\
	echo "events = \\" > "$$tmp" && sed -E 				\
		-e 's|.*/works/(.*)/part_[0-9]+.gz|events/\1 \\|' 	\
		-e 's/=/$$(EQ)/' manifest.txt | uniq >> "$$tmp"; 	\
	if ! cmp -s "$$tmp" remote_targets.mk; then 			\
		mv "$$tmp" remote_targets.mk; 				\
	fi

# warn if the we're within a day after manifest update or 20 days out, then allow the
# user to ctrl+C within a short amount of time? Can I get the manifest time form
# HTTP headers, or will I need to bring in the pull parse pattern after all?
FORCE:
manifest.txt: FORCE
	tmp=$$(mktemp); 							\
	curl -s "https://openalex.s3.amazonaws.com/data/works/manifest" | 	\
		jq -r .entries[].url | sort > "$$tmp"; 				\
	if ! cmp -s "$$tmp" manifest.txt; then 					\
		mv "$$tmp" manifest.txt;					\
	fi

.PHONY: recover
recover:
	$(PYTHON) dump.py $(DUMPFLAGS) $(DATA_DIR) data.sqlite
	cp -r abstracts-embeddings/events ./

.PHONY: clean
clean:
	conda run -n abstracts-search --live-stream python train.py clean
	rm -rf events
	rm -rf $(DATA_DIR)
	rm -rf $(INDEX_DIR)
	rm -f data.sqlite
	rm -f remote_targets.mk
	rm -f oa_jsonl
