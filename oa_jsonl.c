// oa_jsonl.c

// Copyright 2024 Kenny Peng

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#define ASSERT_INCREMENT(ptr, c) do {assert(*(ptr) == c); ++(ptr);} while(0)
#define DEFAULT_SIZE 100

// TODO: give it error handling?

typedef struct OaJsonl {
    char **words;
    int words_size, n_words;
    char *abstract;
    int abstract_size;
} OaJsonl;

static int initial_number_char(char c) {
    return (c >= '0' && c <= '9') || c == '-';
}

static int number_char(char c) {
    return initial_number_char(c) || c == '+' || c == 'e' || c == 'E' || c == '.';
}

static char* advance_space(char *ptr) {
    // '\n' is not acceptable whitespace in JSONL format
    for (; *ptr == ' ' || *ptr == '\t' || *ptr == '\r'; ++ptr);
    return ptr;
}

static char* advance_string(char *ptr, int terminate) {
    ASSERT_INCREMENT(ptr, '"');
    
    // because only '"' and "\"" are relevant, break from normal JSON parsing
    // with a backwards-advancing pointer scheme to achieve the minimal loop
    while (1) {
        if (*ptr == '"') {
            int cnt = 0;
            for (char *tmp = ptr-1; *tmp == '\\'; --tmp) ++cnt;
            if ((cnt & 0x1) == 0) break;
        }
        ++ptr;
    }

    if (terminate) {
        assert(*ptr == '"');
        *(ptr++) = '\0';
    } else {
        ASSERT_INCREMENT(ptr, '"');
    }

    return ptr;
}

static char* advance_composite_open(char *ptr, char open_c) {
    ptr = advance_space(ptr);
    ASSERT_INCREMENT(ptr, open_c);

    // edge case: list or object can be empty, so burn through whitespace
    // to potentially find close_c, even if it's the first value's whitespace
    ptr = advance_space(ptr);

    return ptr;
}

static char* advance_composite_try_close(char *ptr, char close_c) {
    if (*ptr != close_c) {
        return NULL;
    }
    ++ptr;  // close_c
    ptr = advance_space(ptr);
    return ptr;
}

static char* advance_composite_next(char *ptr) {
    if (*ptr == ',') {
        ++ptr;  // ','
    }
    return ptr;
}

static char* advance_value_skip(char *ptr) {
    ptr = advance_space(ptr);
    if (initial_number_char(*ptr)) {  // number
        for (; number_char(*ptr); ++ptr);
    } else if (*ptr == 'f') {  // false
        ptr += 5;
    } else if (*ptr == 't' || *ptr == 'n') {  // true or null
        ptr += 4;
    } else if (*ptr == '"') {  // string
        ptr = advance_string(ptr, 0);
    } else if (*ptr == '{' || *ptr == '[') {  // array or object
        int brackets = (*ptr == '[');
        int braces = (*ptr == '{');
        ++ptr;

        while (brackets || braces) {
            if (*ptr == '"') {
                ptr = advance_string(ptr, 0);
            } else {
                switch (*ptr) {
                    case '[': ++brackets; break;
                    case ']': --brackets; break;
                    case '{': ++braces; break;
                    case '}': --braces; break;
                }
                ++ptr;
            }
        }
    } else {  // not a JSON type (impossible for valid JSON)
        assert(0);
    }
    ptr = advance_space(ptr);
    return ptr;
}

static char* parse_string(char *ptr, char **dst) {
    ptr = advance_space(ptr);
    *dst = ptr + 1;  // string starts after '"'
    ptr = advance_string(ptr, 1);
    ptr = advance_space(ptr);
    return ptr;
}

static char* parse_nullable_string(char *ptr, char **dst) {
    // similar to parse_string, but '"' is not a hard requirement
    ptr = advance_space(ptr);
    if (*ptr == '"') {
        *dst = ptr + 1;  // string starts after '"'
        ptr = advance_string(ptr, 1);
    } else if (*ptr == 'n') {
        *dst = NULL;
        ptr += 4;
    } else {
        assert(0);
    }
    ptr = advance_space(ptr);
    return ptr;
}

static char* parse_name(char *ptr, char **dst) {
    // similar to parse_string, but colon is a hard requirement
    ptr = parse_string(ptr, dst);  // same code here
    ASSERT_INCREMENT(ptr, ':');
    return ptr;
}

// TODO: add checking to hthis
static char* parse_integer(char *ptr, int *val) {
    int neg = 0;
    ptr = advance_space(ptr);
    if (*ptr == '-') {
        neg = 1;
        ++ptr;  // '-'
    }
    *val = *(ptr++) - '0';
    while (*ptr >= '0' && *ptr <= '9') {
        *val *= 10;
        *val += *(ptr++) - '0';
    }
    if (neg) {
        *val = -(*val);
    }
    ptr = advance_space(ptr);
    return ptr;
}

static char* parse_list_open(char *ptr) {
    return advance_composite_open(ptr, '[');
}

static char* parse_list_try_close(char *ptr) {
    return advance_composite_try_close(ptr, ']');
}

static char* parse_list_next(char *ptr) {
    return advance_composite_next(ptr);
}

static char* parse_object_open(char *ptr) {
    return advance_composite_open(ptr, '{');
}

static char* parse_object_try_close(char *ptr) {
    return advance_composite_try_close(ptr, '}');
}

static char* parse_object_next(char *ptr) {
    return advance_composite_next(ptr);
}

OaJsonl* oajsonl_alloc(int words_size, int abstract_size) {
    if (words_size < 0) {
        words_size = DEFAULT_SIZE;
    }
    if (abstract_size < 0) {
        abstract_size = DEFAULT_SIZE;
    }

    OaJsonl* oa = (OaJsonl*)malloc(sizeof(OaJsonl));
    oa->words = (char**)malloc(words_size*sizeof(char*));
    oa->words_size = words_size;
    oa->abstract = (char*)malloc(abstract_size*sizeof(char));
    oa->abstract_size = abstract_size;

    return oa;
}

static void oajsonl_init_abstract(OaJsonl* oa) {
    oa->n_words = 0;
    oa->abstract[0] = '\0';
}

static void oajsonl_add_word(OaJsonl* oa, int idx, char* word) {
    int prev;
    
    if (idx >= oa->words_size) {
        prev = oa->words_size;
        oa->words_size = idx*2;
        oa->words = (char**)realloc(oa->words, oa->words_size*sizeof(char*));
    }

    if (idx >= oa->n_words) {
        prev = oa->n_words;
        oa->n_words = idx+1;
        for (int i = prev; i < oa->n_words; i++) {
            oa->words[i] = NULL;
        }
    }

    oa->words[idx] = word;
}

// TODO: this could be cleaner?
static char* oajsonl_realloc_abstract(OaJsonl* oa, char* ptr) {
    int n = (ptr != NULL) ? ptr - oa->abstract : -1;
    oa->abstract_size *= 2;
    oa->abstract = (char*)realloc(oa->abstract, oa->abstract_size);
    return ptr ? oa->abstract + n : NULL;
}

void oajsonl_build_abstract(OaJsonl* oa) {
    char* tmp = oa->abstract;
    for (int i = 0; i < oa->n_words; i++) {
        if (oa->words[i] == NULL) {
            continue;
        }

        for (char *tmp_2 = oa->words[i]; *tmp_2 != '\0'; ++tmp_2) {
            *(tmp++) = *tmp_2;
            if (tmp >= (oa->abstract + oa->abstract_size)) {
                tmp = oajsonl_realloc_abstract(oa, tmp);
            }
        }

        if (i != oa->n_words-1) {
            *(tmp++) = ' ';
            if (tmp >= (oa->abstract + oa->abstract_size)) {
                tmp = oajsonl_realloc_abstract(oa, tmp);
            }
        }
    }
    *tmp = '\0';
}

char* oajsonl_parse_abstract_inverted_index(OaJsonl *oa, char *ptr, char** abstract) {
    *abstract = NULL;
    
    // edge case: abstract_inverted_index is nullable, so burn through
    // whitespace to potentially find null, even if it's the object's whitespace
    ptr = advance_space(ptr);
    if (*ptr == 'n') {
        ptr += 4;
        ptr = advance_space(ptr);
        return ptr;
    }

    oajsonl_init_abstract(oa);

    char *tmp, *tmp_2;

    for (
        ptr = parse_object_open(ptr);
        (tmp = parse_object_try_close(ptr)) == NULL;
        ptr = parse_object_next(ptr)
    ) {
        char *word;
        ptr = parse_name(ptr, &word);
        
        for (
            ptr = parse_list_open(ptr);
            (tmp_2 = parse_list_try_close(ptr)) == NULL;
            ptr = parse_list_next(ptr)
        ) {
            int idx;
            ptr = parse_integer(ptr, &idx);
            oajsonl_add_word(oa, idx, word);
        }
        ptr = tmp_2;
    }
    ptr = tmp;

    oajsonl_build_abstract(oa);
    *abstract = oa->abstract;

    return ptr;
}

void oajsonl_destroy(OaJsonl* oa) {
    free(oa->words);
    free(oa->abstract);
    free(oa);
}

int read_line(FILE *f, char** line, int line_size) {
    int c = -1, i = 0;
    while (c != '\n') {
        c = fgetc(f);
        if (c == EOF) {
            c = '\n';
        }
        (*line)[i++] = c;

        if (i >= line_size) {
            line_size *= 2;
            *line = (char*)realloc(*line, line_size);
        }
    }
    (*line)[i] = '\0';
    return line_size;
}

int main(){
    int line_size = DEFAULT_SIZE;
    char *line = (char*)malloc(line_size);

    OaJsonl *oa = oajsonl_alloc(-1, -1);

    FILE *f = stdin;
    while (!feof(f)) {
        line_size = read_line(f, &line, line_size);

        // if the file ended with '\n', then the last "line" will be empty
        if (line[0] == '\n') {
            break;
        }

        int drop = 0;
        char *ptr, *tmp, *tmp_2;
        char *id, *title, *lang, *abstract;
        for (
            ptr = parse_object_open(line);
            (tmp = parse_object_try_close(ptr)) == NULL;
            ptr = parse_object_next(ptr)
        ) {
            ptr = parse_name(ptr, &tmp_2);

            if (strcmp(tmp_2, "id") == 0) {
                ptr = parse_string(ptr, &id);
            } else if (strcmp(tmp_2, "title") == 0) {
                ptr = parse_nullable_string(ptr, &title);
            } else if (strcmp(tmp_2, "language") == 0) {
                ptr = parse_nullable_string(ptr, &lang);
                if (!lang || strcmp(lang, "en") != 0) {
                    drop = 1;
                    break;
                }
            } else if (strcmp(tmp_2, "abstract_inverted_index") == 0) {
                ptr = oajsonl_parse_abstract_inverted_index(oa, ptr, &abstract);
                if (!abstract || abstract[0] == '\0') {
                    drop = 1;
                    break;
                }
            } else {
                ptr = advance_value_skip(ptr);
            }
        }
        if (drop) {
            continue;
        }
        ptr = tmp;

        // TODO: handle UTF-16 surrogate pairs instead of outputting JSON
        if (title) {
            printf(
                "{\"id\":\"%s\",\"document\":\"%s %s\"}\n", id, title, abstract
            );
        } else {
            printf(
                "{\"id\":\"%s\",\"document\":\"%s\"}\n", id, abstract
            );
        }
    }

    oajsonl_destroy(oa);
}
