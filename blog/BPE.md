---
layout: doc
title: Tiny LLM gets a real tokenizer
---

# Tiny LLM gets a real tokenizer

*May 18, 2026*

---

A while back I built [tinyLLM](https://github.com/vinayakankugoyal/tinyLLM), in JAX that learns to generate Shakespeare. The version I built first was character-level, i.e. every character got an integer, the vocabulary was 65 tokens, and the model predicted the next character. The problem with that is that "hello" eats 5 tokens.

Time to upgrade to a real tokenizer. This post is about Byte Pair Encoding (BPE): what it does, how I implemented it, and what it taught me.

::: info 📝 Note
This builds on the previous tinyLLM posts ([part 1](/blog/tinyLLM), [part 2](/blog/KVCache)). You don't strictly need them to follow this, but the model and codebase will make more sense if you've seen them.
:::

## Table of Contents

[[toc]]

## The problem with character-level

In part 1 our tokenizer was literally:

```python
sorted_chars = sorted(set(text))
char_to_encoding = {c: i for i, c in enumerate(sorted_chars)}
encoding_to_char = {i: c for i, c in enumerate(sorted_chars)}
```

65 unique characters in Shakespeare, 65 tokens. Simple, reversible, easy to reason about. The problem: every single character costs a token of context. With `CONTEXT_LENGTH=256`, our model can only see the last ~40 words of any prompt. That's nothing. Real LLMs use subword tokenization so a token roughly maps to a syllable or a common word, packing way more text into the same context window.

The standard subword algorithm is **Byte Pair Encoding (BPE)**.

## What BPE actually does

The algorithm is pretty simple:

1. Start with a vocabulary of single characters.
2. Look at every adjacent pair of characters in your training corpus.
3. Find the most common pair. Merge it into a single new token. Add it to the vocabulary.
4. Repeat until your vocabulary is the size you want.

That's it. The first merge might be `("t", "h")` because `th` is everywhere in English. The next might be `(" ", "t")` (space + t). After a few hundred merges, you've got tokens like `" the"`, `" and"`, `"ing"`. Vocabulary grows, but each token covers more characters, so encoded text gets shorter.

::: tip 💡 Intuition: BPE is just learned find-and-replace
A trained tokenizer is nothing more than an ordered list of "replace pair X with new token Y" rules. To tokenize new text, you replay the rules in the exact same order you learned them. That's why training and encoding share so much code. They're doing the same thing, just one builds the rules and the other applies them.
:::

## Splitting text into words first

Before we run BPE, there's one thing we need to handle. You don't actually run BPE on the entire corpus as one long stream. You run it on a list of *words*, separately. Why? Because you don't want a token that spans `"the cat"`. The space between words is information the model should predict, not pre-merge.

But "split into words" is a trap. If you just `text.split()`, you've thrown away all the whitespace and the decoder can never reconstruct the original. I need `decode(encode(x)) == x`. So the whitespace has to live somewhere.

The trick, which GPT-2 also uses, is to attach the leading whitespace to the word that follows it. So `"the cat sat"` becomes `["the", " cat", " sat"]`. The space lives on as a prefix character.

```python
def _text_to_words(text: str) -> list[str]:
    result = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_words = line.split()
        if len(line_words) == 0:
            continue
        if i != 0:
            line_words[0] = "\n" + line_words[0]
        for i in range(1, len(line_words)):
            line_words[i] = " " + line_words[i]
        result.extend(line_words)
    return result
```

## Linked lists for the corpus

Now for the BPE loop. The naive way is to keep words as lists of strings and rebuild them each merge: scan, find pairs, build a new list with the merged pair replaced. That's O(n) per merge step, and it allocates a new list every iteration.

I went with linked lists instead:

```python
class _Node:
    def __init__(self, data: str):
        self.token = data
        self.next = None

    def pairs(self) -> list[tuple[tuple[str, str], _Node]]:
        result = []
        current = self
        while current is not None and current.next is not None:
            result.append(((current.token, current.next.token), current))
            current = current.next
        return result

    def merge(self):
        self.token = self.token + self.next.token
        self.next = self.next.next
```

Every word is a linked list of single-character nodes. To merge a pair, I just concatenate the two strings on one node and skip over the next node. No shifting, no allocation, no copying. The `pairs()` method walks the list and returns each adjacent pair along with the node that starts it, which is exactly what I need to apply merges later.

::: tip 💡 Intuition: Store the node, not the index
The reason `pairs()` returns `(pair, node)` instead of `(pair, index)` is that indices change every time you merge. A merge at position 5 shifts everything after it. Node references don't change. The node at position 5 is still the node at position 5 (well, position 5-ish) after the merge. This is the whole reason linked lists are nice here.
:::

## The training loop

With words-as-linked-lists in hand, the BPE loop fits in your head:

```python
def _bpe(words: list[str], vocab_size: int = 500) -> tuple[list, list]:
    merge_rules = []
    vocabulary = sorted({char for word in words for char in word})
    linked_lists = _words_to_linked_lists(words)

    while len(vocabulary) < vocab_size:
        pair_freqs = defaultdict(list)
        for linked_list in linked_lists:
            for pair, node in linked_list.pairs():
                pair_freqs[pair].append(node)

        most_frequent_pair = max(pair_freqs, key=lambda k: len(pair_freqs[k]))
        merge_rules.append(most_frequent_pair)
        vocabulary.append(most_frequent_pair[0] + most_frequent_pair[1])

        for node in pair_freqs[most_frequent_pair]:
            node.merge()

    return (merge_rules, vocabulary)
```

Initial vocabulary is the set of unique characters. Then, until the vocab hits the target size: count every pair across every word, find the most frequent one, record the merge rule, add the merged string to the vocab, and apply the merge to every recorded occurrence.

Notice `pair_freqs` stores not just the count but the actual list of nodes where each pair appears. That's the linked-list payoff. I already know exactly which nodes to merge, so I don't have to scan the corpus again.

## Encoding new text

Once training is done, the tokenizer holds two things: a `vocabulary` (list of strings, where the index is the token ID) and an ordered list of `merge_rules`. To encode new text:

```python
def encode(self, text: str) -> list[int]:
    words = _text_to_words(text)
    linked_lists = _words_to_linked_lists(words)
    for merge_pair in self.__merge_rules:
        all_pairs = defaultdict(list)
        for linked_list in linked_lists:
            for pair, node in linked_list.pairs():
                all_pairs[pair].append(node)
        for node in all_pairs[merge_pair]:
            node.merge()
    result = []
    for linked_list in linked_lists:
        result.extend(self.__encode_word_list(linked_list))
    return result
```

Split into words, turn each word into a linked list, apply every merge rule in order, then walk the lists and emit each node's integer ID.

::: tip 💡 Intuition: Why merge rules must be applied in order
Training learned `("t","h")` before `("th","e")`. If you applied `("th","e")` first on `"the"`, it would do nothing. There's no `"th"` token yet to combine with `"e"`. The merges *build on each other*, and the order encodes a dependency graph. Replay them out of order and you get garbage.
:::

Decode is trivial. Look up each integer in the vocabulary and concatenate:

```python
def decode(self, encoding: list[int]) -> str:
    return "".join(self.__int_to_token[i] for i in encoding)
```

## Wiring it up

I extracted the tokenizer into its own module (`tokenizer.py`) and added a `--tokenize` step to `tinyLLM.py` that runs before training. The flow is now:

1. `python tinyLLM.py --tokenize --input input.txt` learns the tokenizer from the corpus and saves it to `tokenizer_params.pkl`.
2. `python tinyLLM.py --train` loads the tokenizer, encodes the corpus, and trains the model.
3. `python tinyLLM.py --inference --prompt "..."` loads the tokenizer and model, and generates text.

::: info 📝 Note
Splitting the tokenizer out as a separate step matters because BPE training is slow-ish and doesn't depend on model hyperparameters. You don't want to re-train the tokenizer every time you tweak the embedding dimension. Train it once, save it, reuse it.
:::

## Did it help?

With `vocab_size=500`, my Shakespeare corpus shrinks roughly 3x in token count. Same `CONTEXT_LENGTH=256` now covers ~3x more text. The generated text is qualitatively better too: fewer broken-up words, more believable phrasing.

But honestly the bigger payoff was *understanding what a tokenizer is*. Before this, I had a vague magical view: "BPE produces subwords somehow." After writing it, I know that a tokenizer is just a small ordered list of string-replacement rules learned by frequency, plus a vocabulary table.

If you liked this post please share it with your friends!

You can find the complete implementation [here](https://github.com/vinayakankugoyal/tinyLLM.git).
