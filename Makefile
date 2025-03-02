SHELL := bash
PYTHON := conda run -n abstracts-search --live-stream python

CFLAGS := -O2
BUILDFLAGS :=
DUMPFLAGS :=
TRAINFLAGS :=

abstracts-faiss/index: abstracts-embeddings/data
	$(PYTHON) ./train.py train $(TRAINFLAGS) $< $@

include remote_targets.mk
EQ := =
abstracts-embeddings/data abstracts-embeddings/events &: | $(events)
	$(PYTHON) ./dump.py $(DUMPFLAGS) data.sqlite abstracts-embeddings/data
	cp -r events abstracts-embeddings/

# in theory, wouldn't I need to handle two objects in one line here? I could probably
# update oa_jsonl to handle that?
# consider downloading the files via curl instead of aws s3 cp, even if we keep s3 ls?
EQ := =
events/updated_date$(EQ)% : oa_jsonl | data.sqlite events
	d="s3://openalex/data/works/$(subst events/,,$@)"; 			\
	aws s3 ls --no-sign-request "$$d/" | 					\
		sed -E "s|.* +(.*)|$$d/\1|" | 					\
		xargs -I % -- aws s3 cp --no-sign-request % - | 		\
		gunzip | ./oa_jsonl | mbuffer -q -t -m 16G | 			\
		$(PYTHON) ./build.py $(BUILDFLAGS) data.sqlite
	touch $@

oa_jsonl: oa_jsonl.c
	$(CC) $(CFLAGS) -o $@ $<

data.sqlite:
	$(PYTHON) -m utils.table_utils $@

events:
	mkdir events

# consider using curl for the manifest then learning to use a JSON query tool
# warn if the we're within a day after manifest update or 20 days out, then allow the
# user to ctrl+C within a short amount of time? Can I get the manifest time form
# HTTP headers, or will I need to bring in the pull parse pattern after all?
FORCE:
remote_targets.mk: FORCE
	tgts="$$(mktemp)"; 							\
	echo "events = \\" >> $$tgts; 						\
	for p in $$( 								\
		aws s3 ls --no-sign-request "s3://openalex/data/works/" | 	\
		sed -E "s|.* +(.*)|\1|" 					\
	); do 									\
		if [[ "$$p" == "manifest" ]]; then 				\
			continue; 						\
		fi; 								\
		echo "events/$${p%/} \\" | sed -e "s/=/\$$\(EQ\)/" >> $$tgts; 	\
	done; 									\
	cmp -s "$$tgts" $@; 							\
	if [[ $$? != 0 ]]; then 						\
		mv $$tgts $@; 							\
	fi

.PHONY: recover
recover:
	$(PYTHON) ./dump.py $(DUMPFLAGS) abstracts-embeddings/data data.sqlite
	cp -r abstracts-embeddings/events ./

.PHONY: clean
clean:
	conda run -n abstracts-search --live-stream python train.py clean
	rm -rf events
	rm -rf abstracts-embeddings/data
	rm -rf abstracts-index/index
	rm -f data.sqlite
	rm -f remote_targets.mk
	rm -f oa_jsonl
