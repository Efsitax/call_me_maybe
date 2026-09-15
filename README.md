*This project has been created as part of the 42 curriculum by kugurlu.*

# call me maybe

## Description

**call me maybe** is a function-calling tool that translates natural-language
questions into structured function calls. Given a question like *"What is
the sum of 2 and 3?"*, it does not answer the question directly — it decides
**which function** should be called (`fn_add_numbers`) and **what arguments**
it should receive (`{"a": 2.0, "b": 3.0}`).

The core constraint of the project is that this decision must come from an
LLM (`Qwen/Qwen3-0.6B` by default), and the LLM's raw output must be turned
into 100% valid, schema-compliant JSON — without ever prompting the model
and hoping it produces the right format. Instead, the project implements
**constrained decoding**: at every generation step, the model's logits are
masked so that only tokens which keep the output structurally valid can be
selected. The model still decides *what* to say; the code only decides
*what shapes of answer are legal*.

The tool reads a list of available functions (`functions_definition.json`)
and a list of natural-language prompts (`function_calling_tests.json`), and
produces `function_calling_results.json`: one `{prompt, name, parameters}`
entry per prompt.

## Instructions

### Requirements

- Python 3.10+ (developed and tested on 3.14)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package copied alongside `src/` (already included in this
  repository)

### Setup

```sh
make install   # uv sync
```

### Running

```sh
make run
```

This reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`.

Custom paths and a different model can be passed through `ARGS`:

```sh
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json"
make run ARGS="--model Qwen/Qwen3-1.7B"
```

Or directly, without `make`:

```sh
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>] [--model <hf_model_id>]
```

### Other Makefile targets

```sh
make debug        # run under pdb
make lint         # flake8 + mypy (subject's required flags)
make lint-strict   # flake8 + mypy --strict
make clean        # remove caches, data/output/, and .venv
```

## Algorithm Explanation: Constrained Decoding

Every generation step in this project follows the same core loop
(`src/decoding.py`):

1. Ask the model for logits over the full vocabulary:
   `model.get_logits_from_input_ids(input_ids)`.
2. Compute the set of token ids that are **still valid** at this point —
   the definition of "valid" depends on what is being generated (see
   below).
3. Set every logit outside that set to `-inf` (`_mask_logits`).
4. Take `argmax` over the masked logits. Because invalid tokens are `-inf`,
   they can never win — but among the valid ones, the model's own ranking
   is left untouched. The model still decides *which* legal token it
   prefers; the code only removes the illegal options from the table.
5. Append the chosen token and repeat until a stopping condition is met.

Softmax is deliberately never applied: `argmax(logits) == argmax(softmax(logits))`
because `exp()` is monotonic, so normalizing into probabilities would not
change which token wins — it would just be wasted computation over a
~150,000-entry vocabulary at every step.

Three different notions of "valid" are used, one per value type:

- **Closed-set selection** (`select_candidates`) — used for the function
  name and for booleans. A token is valid if appending it still leaves at
  least one candidate (e.g. a real function name) as a possible completion
  (`_surviving_candidates`). Decoding stops the moment the accumulated text
  exactly equals one of the candidates.
- **Numbers** (`generate_number`) — there is no fixed candidate list (a
  number can be almost anything), so validity is defined by grammar
  instead: only digits, at most one `.`, at most one leading `-`
  (`_is_valid_number_prefix`). The model is free to write any digit
  sequence that stays inside this grammar.
- **Strings** (`generate_string`) — almost fully open-ended; every token is
  valid except a literal `"`, which signals the JSON string should close.

## Design Decisions

- **Public/private function split.** Public functions (no leading
  underscore) are the safety boundary: they wrap their logic in
  `try/except` and fail predictably (`sys.exit(1)` for setup failures, or a
  clear exception during generation). Private functions are kept "pure" —
  they raise naturally and are never called directly from `main()`.
- **Dependency injection over globals.** `model` and the vocabulary lookup
  (`id_to_text`) are always passed explicitly into functions rather than
  read from module-level state. This keeps every function testable in
  isolation and makes the multi-model support possible.
- **Parameters are extracted as a Python call, not as prose.** The
  parameter prompt shows the function as a Python signature with its
  docstring and asks for "the Python call for this request"; the
  assistant's turn is then pre-seeded with the call written up to the
  argument being generated, e.g.
  `fn_substitute_string_with_regex(source_string="Programming is fun", regex="[aeiou]", replacement="`.
  Two things fall out of this framing for free. First, a code context
  makes the model write argument *values* rather than the words that
  describe them: for "replace vowels with asterisks" it writes `[aeiou]`
  and `*`, where an earlier prose-style prefill
  (`The 3rd parameter's value is "`) copied the literal words
  "vowels" and "asterisks" from the question. Second, every earlier
  argument is visible while generating the next one, so the model
  never assigns the same fragment to two parameters. `"` still serves
  as the stop signal for strings.
- **Escape sequences are excluded from the string grammar, not
  decoded afterwards.** Because the prefill opens a double-quoted Python
  string, the model wants to write what a Python string body would
  contain — `C:\\Users\\john` for a Windows path. Rather than
  post-process the output, `generate_string` masks every token that
  contains `\\` (two backslashes), exactly the way `"` is treated as a
  stop signal: the grammar says a value is raw text, and the model then
  picks its next-best token, `C:\Users\john`. The value written is
  still entirely the model's; the code only removed one shape from the
  table. The cost is that a value which legitimately contains two
  consecutive backslashes (a UNC path, a regex matching a literal
  backslash) cannot be produced.
- **Question placed last in the prompt.** Early versions put the question
  before the instructions; the model would sometimes echo nearby prompt
  words (e.g. output the literal word "name" for a parameter called
  `name`) instead of answering. Moving the question to the very end of the
  prompt fixed this.
- **Grounding-guard retries instead of post-processing.** Several
  generation bugs (see below) share a pattern: the model's own output is
  a good format but statistically wrong. Rather than patch the output
  after the fact, `generate_number` and `generate_string` validate the
  model's own result against the question text and, if it looks
  inconsistent, discard it and let the model **generate again** with a
  different first token excluded. The model always produces the final
  value; the code never injects one.
- **Never dropping a prompt.** If a single prompt raises an exception,
  `main()` still appends a placeholder result for it instead of skipping
  it. Function-calling graders typically compare result lists to
  correction lists by position; dropping even one entry silently
  misaligns every result that follows it.
- **Vocabulary read from `tokenizer.json`, not the model's native vocab
  file.** `tokenizer.json`'s `model.vocab` field has the same
  `{token: id}` shape regardless of whether the underlying tokenizer is
  BPE, Unigram, or WordPiece — unlike the native vocab file, whose format
  varies per model (`vocab.json` for GPT-2/Qwen-style tokenizers,
  `tokenizer.model`, a binary SentencePiece file, for Llama-style ones).
  Reading from the unified field is what makes multi-model support
  possible without per-tokenizer branching.

## Performance Analysis

Graded with the `moulinette` tool against both its public and private test
sets (11 prompts each):

| Set | Result |
|---|---|
| Public | 10/11 (90.9%) — PASSED |
| Private | 10/11 (90.9%) — PASSED |

- **JSON validity: 100%.** Every run produces a fully parseable output
  file; constrained decoding guarantees this structurally, and the
  never-drop-a-prompt design guarantees it even on a per-prompt failure.
- **Speed.** All 11 public-set prompts complete in ~11 seconds on Apple
  Silicon (measured with `time uv run python -m src`), comfortably inside
  the 5-minute budget.
- **Where accuracy is lost.** The two remaining failures are the regex
  for "replace all numbers" (the model writes the literal `34` from the
  question instead of `\d+`) and a template whose correct value contains
  a `"` (`Say "hello" to {name}`, where the model switches to single
  quotes). Both are discussed below.

### Multi-model results

The same code, unmodified, run against several models via `--model`:

| Model | Tokenizer family | Result |
|---|---|---|
| `Qwen/Qwen3-0.6B` (default) | BPE | Public 10/11, Private 10/11 |
| `Qwen/Qwen3-1.7B` | BPE | Public 10/11, Private 9/11 |
| `Qwen/Qwen2.5-0.5B-Instruct` | BPE | Public 11/11 (PERFECT) — it also writes `\d+` for "all numbers" |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | SentencePiece-derived | Runs end to end, no crash; lower accuracy |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | BPE | Runs, but function selection collapses to one answer |
| `gpt2` | BPE (base, not instruction-tuned) | Runs, but collapses to a fixed positional answer regardless of the question |

Two things are worth separating here. **Architectural generality is
proven**: every model above loads, and the pipeline never crashes,
regardless of chat-template support, stop-token naming, or tokenizer
format. **Output quality is not**, and isn't expected to be — GPT-2 and
SmolLM2 fail because they lack (or are too small for) instruction-following
ability, not because of a bug; this was confirmed by testing two
completely different decoding strategies (name-based and index-based
selection) against GPT-2, both of which collapsed to a fixed positional
answer regardless of the question. Interestingly, a bigger model
(Qwen3-1.7B) did not automatically outperform the default 0.6B model on
the private set with the same prompting strategy — model *scale* alone
does not fix the remaining failure category described below.

## Challenges Faced

- **A silent alignment bug that looked like a huge accuracy problem.**
  Early on, any prompt that raised an exception was simply skipped, with
  nothing appended to the results list. Because moulinette compares
  results to corrections positionally, a single failure silently shifted
  every later comparison — a private-set score of 1/11 (9%) turned out to
  be almost entirely this bug, not real inaccuracy. Fixed by always
  appending a placeholder result on failure (see Design Decisions).
- **A decoded special token that was `""`, not `None`.** `model.decode([id])`
  for `<|im_end|>` returned an empty string rather than `None`, which made
  the constrained loop append nothing forever — an infinite loop. Fixed by
  explicitly treating an empty decode as `None`.
- **A repeat-guard that broke legitimate repeated characters.** An early
  version of `generate_string` stopped generation after seeing the same
  token twice in a row, meant to prevent runaway loops. It also broke
  answers like `"aaaaa"`. Removed; a stop-token/quote-based stop was
  enough on its own.
- **A quote character used as a stop signal collided with quoted
  content.** `generate_string` stops on `"` — but if the *correct* answer
  itself legitimately contains a `"` (e.g. `Say "hello" to {name}`), the
  mechanism cuts the string off in the middle. A partial fix
  (`split('"')[0]` instead of discarding the whole token) recovered
  content that shared a token with the closing quote, but a value with an
  embedded quote in the middle remains an open limitation of this design.
- **Digit concatenation and wrong-digit errors in `generate_number`.**
  Testing against multiplication/addition prompts revealed the model
  sometimes continued past a complete, correct number (`"2"` → `"23"`), or
  produced an entirely wrong first digit. Both were fixed by the
  grounding-guard mechanism: complete numbers that exactly match a literal
  number in the question force an immediate stop (fixing the
  concatenation case), and numbers that don't match any literal number
  trigger a retry with that first token excluded (fixing the wrong-digit
  case) — up to a bounded number of attempts.
- **A truncated string that still passed a naive "is it a substring"
  check.** `"home/user/data.json"` is technically a substring of
  `"/home/user/data.json"` — so a first grounding check based on plain
  substring containment silently accepted the truncated (wrong) answer.
  Fixed with a boundary check: a match is only accepted if it isn't
  immediately preceded or followed by a "continuation" character
  (letter, digit, `/`, `.`, `-`).
- **A different tokenizer file format per model family.**
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` crashed with a `UnicodeDecodeError`
  because its native vocab file (`tokenizer.model`) is a binary
  SentencePiece file, not the JSON that Qwen/GPT-2-style tokenizers use.
  Fixed by reading the vocabulary from `tokenizer.json`'s `model.vocab`
  field instead, which has the same shape for every tokenizer backend
  (see Design Decisions).
- **Values that must be derived, not extracted.** Two tests ask for a
  value that never appears in the question at all — a regex synthesized
  from a concept ("vowels" → `[aeiou]`) and a symbol described by a word
  ("asterisks" → `*`). The grounding guard cannot help here by design
  (it only rejects values *absent* from the question when *some other
  grounded* value existed to retry toward), so this was investigated
  directly. Several approaches were tested and rejected first:
  - A general (non-answer-specific) hint added to the prose prompt had
    no measurable effect on either Qwen3-0.6B or Qwen3-1.7B.
  - Letting the model reason in an unconstrained `<think>` block before
    answering solved the "numbers → `\d+`" case on Qwen3-1.7B (and
    revealed the empty `<think></think>` prefill we use for speed was
    actively suppressing this reasoning), but it also *lost* previously
    correct extractions elsewhere (the model started copying nearby
    words into fields that used to be extracted correctly), and provided
    no benefit at all on the 0.6B default model.
  - An explicit "should this be copied or derived?" routing question,
    asked before generation, scored 3/10 on a mixed test set — worse than
    guessing, because the model over-applies "derived" as a default.
  - A prompt instruction that maps the literal words "asterisks" or
    "NUMBERS" to their expected outputs was deliberately **not**
    implemented, even though it would pass these two specific tests —
    it is answer-specific to this exact wording, would not survive a
    differently-worded test set, and is exactly the kind of heuristic
    the subject explicitly rules out for this project.

  What finally worked was changing the *framing* rather than adding
  information: presenting the function as a Python signature and
  pre-seeding the assistant with the call written up to the current
  argument (see Design Decisions). In a code context the model's own
  priors take over — it has seen `re.sub(r"[aeiou]", "*", s)` far more
  often than a sentence describing it — and it produced `[aeiou]` and
  `*` on both Qwen3-0.6B and Qwen3-1.7B with no answer-specific text
  in the prompt. The same change first *broke* the Windows-path test:
  the model, now writing a Python string body, correctly escaped the
  backslashes (`C:\\Users\\john`). Priming the string with a raw-string
  prefix (`r"`) was tested and made everything worse (the model started
  inventing leading `\\` and capitalising names). Decoding the body as
  a Python literal with `ast.literal_eval` also restored the test, but
  it rewrites the model's text in code, which is the kind of
  intervention this project avoids. The fix kept was a grammar rule:
  tokens containing `\\` are masked out while generating a string, and
  the model itself then writes the single-backslash path. Two cases remain
  open: `\d+` for "all numbers" (the model still prefers the literal
  `34` from the question), and a template value that itself contains
  `"` — the model switches to `'hello'` rather than emitting `\"`, and
  an escape-aware stop rule was tested and changed nothing because the
  model never chooses the backslash in the first place.

## Testing Strategy

- **`moulinette`**, a grading tool with its own public/private test sets
  and ground-truth corrections, was used throughout as the primary source
  of truth — self-assessment by eye is unreliable at this scale, and
  moulinette's positional comparison is exactly what exposed the
  alignment bug above.
- **Token-by-token debug scripts** were written ad hoc to trace exactly
  which candidate tokens the model was ranking highest at a failing
  decoding step (e.g. showing the top-5 masked logits at each step), which
  is how the digit-concatenation and truncation bugs were diagnosed rather
  than guessed at.
- **Cross-model smoke tests** (`--model <other model>`) were used to
  validate that no part of the pipeline silently depended on
  Qwen3-specific token names, chat-template syntax, or tokenizer file
  format.
- **Continuous `mypy --strict` and `flake8`** were run after every change,
  not just before submission.
- Manual edge-case prompts (empty strings, large numbers, ambiguous
  phrasing) were used while iterating on the prompt design, per the
  subject's recommendation.

## Example Usage

```sh
$ make run
What is the sum of 2 and 3? -> fn_add_numbers parameters: {'a': 2.0, 'b': 3.0}
Greet shrek -> fn_greet parameters: {'name': 'shrek'}
Reverse the string 'hello' -> fn_reverse_string parameters: {'s': 'hello'}
```

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

Running against a different model:

```sh
uv run python -m src --model Qwen/Qwen3-1.7B
```

## Bonus Part

- **Support for multiple LLM models beyond Qwen3-0.6B.** No token id,
  chat-template string, or vocab file format is hardcoded anywhere in
  `src/`. `find_stop_token_id` tries several known end-of-turn token names
  in order (ChatML, GPT-2-style, Llama/Mistral-style); chat-template
  wrapping is skipped automatically for models that don't support it; and
  the vocabulary is read from a tokenizer-agnostic field (see Design
  Decisions). Verified end-to-end against six models across three
  different tokenizer/template families (see Performance Analysis).
- **A from-scratch tokenizer** (`src/tokenizer.py`), avoiding
  `model.encode`/`model.decode` entirely and relying only on
  `get_logits_from_input_ids` and `get_path_to_vocab_file` as the subject
  requires for this bonus:
  - `decode`: rebuilds text from token ids using the GPT-2-style
    byte-to-unicode mapping, validated against the full ~150,000-entry
    real vocabulary with zero mismatches, plus multi-token Unicode
    (Turkish) sentences.
  - `encode`: a full byte-level BPE implementation — pre-tokenization
    regex (adapted from the model's own `tokenizer.json`), iterative
    pairwise merging by learned rank, and vocabulary lookup — validated
    against 16 real test sentences with zero mismatches.
  - Both are public functions, and `select_candidates` accepts an
    optional `encode_fn` parameter demonstrating that the custom
    tokenizer's `encode` can be swapped in for `model.encode` inside the
    constrained-decoding loop with no change in behavior.

## Resources

- [Hugging Face — Tokenizers documentation](https://huggingface.co/docs/tokenizers)
- [Hugging Face — `tokenizer.json` file format](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with
  Subword Units* (2016) — the original BPE-for-NLP paper the byte-level
  tokenizer in this project is based on.
- [Qwen3 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B)
- Guided/constrained generation background: the general idea of masking
  logits to a grammar at each decoding step (as opposed to prompting and
  hoping) is the same principle behind libraries like
  [Outlines](https://github.com/dottxt-ai/outlines) and JSON-schema-guided
  decoding — used here only as conceptual background; none of that
  library's code is used, per the subject's constraints.

### How AI was used

Claude (Claude Code) was used as a mentor and pair-programmer throughout
this project, under an explicit agreement: all code was written by the
author, with Claude explaining concepts, reviewing each change, running
`mypy`/`flake8`/the test pipeline, and pointing out exactly where and why
something was wrong rather than rewriting it. Concretely, Claude was used
for:

- Explaining background concepts on request (tokenization/BPE, logits vs.
  probabilities, softmax, attention, why `argmax` doesn't need softmax or
  log-probabilities) as they came up during implementation.
- Reviewing each function against the codebase's own conventions
  (public/private split, type hints, docstrings) and flagging concrete
  bugs with an explanation (e.g. a self-referential list append,
  `else if` instead of `elif`, a renamed function whose call site wasn't
  updated) rather than silently fixing them.
- Designing and running the debugging methodology described in Testing
  Strategy — building the `moulinette`-based grading loop, and writing
  ad hoc token-by-token trace scripts to diagnose specific failures
  (digit concatenation, truncated strings) instead of guessing at fixes.
- Proposing and empirically testing the grounding-guard design, the
  cross-model architecture changes, and the several rejected approaches
  documented in Challenges Faced (bounded-think, model-scale, routing) —
  including refusing to implement a prompt-based hardcoded mapping for
  the two unsolved test cases, on the grounds that it would violate the
  subject's explicit rule against heuristics and would not generalize to
  a differently-worded test set.
- Writing this README from the project's actual git history, code, and
  test results.
