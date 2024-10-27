EQ := =

.PHONY: all
all: abstracts-embeddings/data

encode_rule := 											\
  <TARGET> : ; 											\
  conda run -n abstracts-search 						\
  python ./build.py 									\
  s3://openalex/data/$$(subst .parquet,.gz,$$@) $$@

include remote_targets.mk

abstracts-embeddings/data: works
	conda run -n abstracts-search python ./encode.py $< $@

# TODO: check for invalidation?
# mcli alias set publics3 https://s3.amazonaws.com "" ""
remote_targets.mk: Makefile  # emits works as a target
	# nice leaning-toothpick syndrome
	tmp_parts=$$(mktemp); \
	tmp_rule=$$(mktemp); \
	printf "works: " >> $$tmp_rule; \
	for p in $$( \
		mcli ls --recursive publics3/openalex/data/works | \
		sed -e 's/.* \(.*\)/\1/' \
	); do \
		if [[ $$p == "manifest" ]]; then continue; fi; \
		p_target_esc=$$( \
			echo $$p | sed -e 's|^|works/|' -e "s/=/\$$\(EQ\)/" -e 's/.gz/.parquet/' \
		); \
		printf "\$$(subst <TARGET>,$$p_target_esc,\$$(encode_rule))\n" >> $$tmp_parts; \
		printf "$$remote_targets\\\\\n  $$p_target_esc" >> $$tmp_rule; \
	done; \
	printf "\n" >> $$tmp_rule; \
	cat $$tmp_rule > $@; \
	cat $$tmp_parts >> $@

.PHONY: clean
clean:
	rm -rf works
	rm -rf abstracts-embeddings/data
	rm remote_targets.mk
