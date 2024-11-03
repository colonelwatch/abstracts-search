SHELL := bash  # TODO: lift this requirement?
MC := mc  # TODO: make aws cli an option?
GZIP := pigz
BUILDFLAGS := --model-name all-MiniLM-L6-v2
CFLAGS := -O2

# Rules cannot have equal signs in the target, so this is a workaround
EQ := =

abstracts-embeddings/data: works
	conda run -n abstracts-search --live-stream python ./encode.py $< $@

oa_jsonl: oa_jsonl.c
	$(CC) $(CFLAGS) -o $@ $<

# A "one-line" rule for getting a parquet from a remote gz, used in remote_targets.mk
encode_rule := 								\
  <TGT> : oa_jsonl; 						\
  $(MC) cat publics3/openalex/data/$$(subst .parquet,.gz,$$@) | 	\
  $(GZIP) -d | ./oa_jsonl | 						\
  conda run -n abstracts-search --live-stream python ./build.py 	\
  $(BUILDFLAGS) $$@

include remote_targets.mk

# mcli alias set publics3 https://s3.amazonaws.com "" ""
# Creates individual rules for each remote updated=XXXX-XX-XX/part-XXX.gz file and a
# rule with all the targets as prereqs. Each rule is an instance of encode_rule.
remote_targets.mk: Makefile  # emits works as a target
	set -e; 								\
	parts=$$(mktemp); 							\
	rule=$$(mktemp); 							\
	printf "works: " >> $$rule; 						\
	for p in $$(								\
		$(MC) ls --recursive publics3/openalex/data/works | 		\
		sed -e 's/.* \(.*\)/\1/'					\
	); do 									\
		if [[ $$p == "manifest" ]]; then 				\
			continue; 						\
		fi; 								\
		tgt="works/$${p/%.gz/.parquet}"; 				\
		printf "\$$(subst <TGT>,$$tgt,\$$(encode_rule))\n" >> $$parts; 	\
		printf "\\\\\n    $$tgt" >> $$rule; 				\
	done; 									\
	printf "\n" >> $$rule; 							\
	cat $$rule | sed -e "s/=/\$$\(EQ\)/" > $@; 				\
	cat $$parts | sed -e "s/=/\$$\(EQ\)/" >> $@

.PHONY: clean
clean:
	rm -rf works
	rm -rf abstracts-embeddings/data
	rm -f remote_targets.mk
	rm -f oa_jsonl
