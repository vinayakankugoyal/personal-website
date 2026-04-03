---
layout: doc
---

# Tiny LLM

*March 30, 2026*

---

I have been reading about inference scaling and the term KV cache comes up a lot, like a loooot. I wanted to learn how it really worked and there is no other way than to implement it from scratch. So in a coffee induced psychosis (and gastritis) I wrote it from scratch (aka not using Claude Code[^1]) and tried to implement all the pieces that helped forward my understanding. I showed what I built to a friend (shout out skainswo!) who said this would make a great blog post. As I started writing the blog post I realized it would be too long and so this is part [1/2] that covers multi-head attention. Part [2/2] will cover KV caching.

## Table of Contents

[[toc]]

## Pre-requisite

I highly recommend using `nix` with this shell.nix config. This should install everything you need.

```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShellNoCC {
    packages = [
        (pkgs.python3.withPackages (ps: [
            ps.jax
            ps.optax
        ]))
    ];
}
```

::: info 📝 Note
If VS Code is not playing nice with this then do the following:

1. In the VS Code terminal run `nix-shell` command
2. Then run `which python`
3. `CMD+SHIFT+P` and type `Python: Select Interpreter` then select `Enter interpreter path...` and add the path from 2.
:::

## Let's go

With the setup out of the way let's jump right into it!

I believe in keeping things simple and sometimes using too many fancy libraries abstracts too many of the details, so I will try to implement as much of the multi-head attention as possible. However, wherever it makes sense to use a library I'll do so because the goal is to develop an intuition about multi-head attention.

I recommend downloading the data from here:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Let's first load the data. Since this is small enough we can just load it into memory.

```python
with open("./input.txt", "r") as f:
    text = f.read()

total_chars = len(text)
vocabulary = sorted(set(text))
vocabulary_size = len(vocabulary)

print(f"{vocabulary_size=}")
print(f"{total_chars=}")
print(f"{vocabulary=}")
```

Running the above or similar should give you something like:

```
vocabulary_size=65
total_chars=1115394
vocabulary=['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
```

Ok that's interesting, we have a few special characters but only 1 number!

Next step is tokenization, that is breaking our data into small chunks that is going to be processed by the LLM. There are many tokenizers out there[^2] but for this exercise I will just use each character as a token (I like to keep things simple!).

The input to the neural network is usually a number so we must convert the characters into numbers somehow. Let's create 2 mappings, one that takes a character and returns a number and another that takes a number and gives its corresponding character in our vocabulary. This will be our encoding and decoding. Remember we sorted the set of all characters before, so now we can use the index of each character as its encoding and reverse that to decode them.

```python
char_to_encoding = {c: i for i, c in enumerate(vocabulary)}
encoding_to_char = {i: c for i, c in enumerate(vocabulary)}

def encode(text: str) -> list[int]:
    return [char_to_encoding[c] for c in text]

def decode(encoded_text) -> str:
    return "".join([encoding_to_char[i] for i in encoded_text])

print(encode("hello"))

print(decode([46, 43, 50, 50, 53]))
```

This should print something like:

```
[46, 43, 50, 50, 53]
hello
```

Now let's encode the entire data and initialize a numpy array for it.

```python
import jax

encoded_text = jax.numpy.array(encode(text))

print(f"{encoded_text.shape=}")
print(f"{encoded_text[:10]=}")
```

This should show something like:

```
encoded_text.shape=(1115394,)
encoded_text[:10]=Array([18, 47, 56, 57, 58,  1, 15, 47, 58, 47], dtype=int32)
```

Now that we have our data encoded we can break it into 2 parts: one for training and one for testing. How you split this is highly dependent on the data. For our use case we will split it 90/10, 90% for training and 10% for testing.

```python
TRAIN_SPLIT = 0.9
split_idx = int(len(encoded_text) * TRAIN_SPLIT)
train_data = encoded_text[:split_idx]
test_data = encoded_text[split_idx:]

print(f"{encoded_text.shape=}")
print(f"{train_data.shape=}")
print(f"{test_data.shape=}")
print(f"{encoded_text[:10]=}")

```

The next thing we need before we get into implementation is a way to chunk or batch this data. There are libraries you can use for this but I implemented this myself.

We will need to make 2 major decisions here:

1. What is the batch size, i.e. how many samples are in a batch?
2. What is the size of each sample? This will be our context length.

I picked BATCH_SIZE=32 and CONTEXT_LENGTH=128. This means in each batch we will have 32 samples and the length of each sample will be 128.

Let's write a `get_batch` function that takes our data and returns a tuple `(inputs, outputs)`:

```python
def get_batch(data, key):
    starting_positions = jax.random.randint(
        key, (BATCH_SIZE,), 0, len(encoded_text) - CONTEXT_LENGTH - 1
    )

    stacked_starting_postions = starting_positions[:, None]

    indices = stacked_starting_postions + jax.numpy.arange(CONTEXT_LENGTH + 1)

    # this is going to be (BATCH_SIZE, CONTEXT_LENGTH + 1)
    stacked = data[indices]

    # take all the batches but skip the last element of each batch so that
    # we get (BATCH_SIZE, CONTEXT_LENGTH)
    inputs = stacked[:, :-1]

    # take all the batches but skip the first element of each batch so that
    # we get (BATCH_SIZE, CONTEXT_LENGTH)
    outputs = stacked[:, 1:]

    return (inputs, outputs)
```

Ok that can be a lot if you are new to `numpy`. So here is the run down.

We start with `starting_positions` which looks something like:

```
[1, 22, 20, 15, 3, 19, 13, 9, 21, 5 ...]
```

We then "stack" them.

```
[[1],
 [22],
 [20],
 [15],
 ...
]
```

Then we have `jax.numpy.arange(CONTEXT_LENGTH + 1)` which looks like:

```
[0, 1, 2, 3, 4, ..128]
```

Now we add the two and with the magic of broadcasting we get:

```
[
 [1, 2 ,3, 4, ...129]
 [22, 23, 24, 25, ...150]
 [20, 21, 22, 23, ...148]
 [15, 16, 17, 18, ...143]
 ...
]
```

This gives us all the indices we want and so when we slice the encoded text we select 32 batches of 129 character sequences starting at different indices.

::: info 📝 Note
I know that can be a lot, so if you aren't too worried about performance you could write this as:

```python
def get_batch(data, rng_key):
    starting_positions = jax.random.randint(rng_key, (BATCH_SIZE,), 0, len(data) - CONTEXT_LENGTH - 1)

    inputs = []
    outputs = []

    for sp in starting_positions:
        input_data = data[sp:sp + CONTEXT_LENGTH + 1]
        output_data = input_data[1:]

        inputs.append(input_data[:-1])
        outputs.append(output_data)

    return (jax.numpy.stack(inputs), jax.numpy.stack(outputs))
```
:::


::: tip 💡 Intuition: What are inputs and outputs?
Let's say for now the context length is 5. Then if a single input sample is:

```
[46, 43, 50, 50, 53] # hello
```

Its corresponding output would be: 

```
[43, 50, 50, 53, 1] # ello<space> 
```

What we are learning is how to predict the next token, so we shift our input to the left by 1 and that becomes our output.
:::


Ok now we have inputs and corresponding outputs. We are now ready to train.

The first step in the LLM would be to take the tokens and convert them to embeddings. We will use 2 kinds of embeddings:
1. Token embedding
2. Positional embedding

**Token embedding** will take our token and project it to a high dimensional space.

**Positional embedding** will represent the position of the token in the sequence.

Let's write a function to embed the token:

```python

params = {}

rand_key, subkey = jax.random.split(rand_key)
# mapping of each token to what its vector representation is
embedding = jax.random.normal(subkey, (VOCABULARY, EMBED_DIM)) * 0.02

params["token_embedding"] = embedding

rand_key, subkey = jax.random.split(rand_key)
# mapping of context_length to what something at a postion looks like
positional_embedding = jax.random.normal(subkey, (CONTEXT_LENGTH, EMBED_DIM)) * 0.02

params["positional_embedding"] = positional_embedding

def embed(params, inputs):
    return params["token_embedding"][inputs] + params["positional_embedding"]
```

::: tip 💡 Intuition: Why Embeddings?
Think about it this way: the characters 'a' and 'b' are numerically just 1 apart (indices 39 and 40 in our vocabulary), but semantically they're not "similar" in any meaningful way. Meanwhile, 'a' and 'A' are 26 apart but represent the same letter!
In encoding we converted the characters to numbers, but raw numbers don't capture meaning. By projecting each token into a high-dimensional space (our `EMBED_DIM`), we give the model room to *learn* its own notion of similarity. After training, tokens that behave similarly in context will end up closer together in this embedding space.
:::

::: tip 💡 Intuition: Why Positional Embeddings?
What's the difference between "dog bites man" and "man bites dog"?

To you and me, everything! But to vanilla attention, absolutely nothing. Attention computes relationships between tokens using dot products, and dot products don't care about order. It's just a way of asking "how similar are these two things?" The sentence is treated like a bag of words.

With positional embeddings we are telling the model "this token appeared at position 3". Now "dog" at position 0 looks different from "dog" at position 2, and the model can learn that a word's position in the sentence matters.
:::

Now let's write a function called `forward` that will take our input and call the embed function on it:

```python
def forward(params, inputs):
    x = embed(params, inputs)

    return x

inputs, outputs = get_batch(train_data, rand_key)

print(f"{inputs.shape=}")
print(f"{outputs.shape=}")

y = forward(params, inputs)

print(f"{y.shape=}")

```

Running what we have should print something like:

```
inputs.shape=(32, 128)
outputs.shape=(32, 128)
y.shape=(32, 128, 128)
```

Nice! We have the first parameters of our model and it's actually doing something!

## Attention

Next we will write the attention function. This is actually the easiest part (IMHO).

Attention is essentially just[^3]:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

Now let's implement that:

```python
def attention(params, inputs):
    Q = inputs @ params["W_q"]
    K = inputs @ params["W_k"]
    V = inputs @ params["W_v"]

    # attention scores
    attention_score = Q @ K.transpose(0, 2, 1)

    # scale attention scores otherwise gradients will vanish
    attention_score_scaled = attention_score / (EMBED_DIM**0.5)

    # causal mask so that we only look at the previos tokens
    causal_mask = jax.numpy.triu(
        jax.numpy.full((CONTEXT_LENGTH, CONTEXT_LENGTH), -jax.numpy.inf), k=1
    )

    attention_score_scaled_masked = attention_score_scaled + causal_mask

    attention_weights = jax.nn.softmax(attention_score_scaled_masked)

    weighted_sum = attention_weights @ V

    return weighted_sum @ params["W_o"]
```

That's it!

::: tip 💡 Intuition: What are Q, K, V?
This concept comes from the database and search engine world. Imagine you are searching for something in a search engine (your ad here :p). You have a **Query** (Q), what you're looking for: "get better jiu-jitsu fast now please". The search engine has **Keys** (K) that describe what they hold: "ways to get better at karate", "ways to get better at jiu-jitsu", "ways to get better at talking to the person you like". The search engine has **Values** (V), the actual content you'd get if you picked that key.

When we compute `Q @ K^T`, we're asking "how relevant is each token's Key to my Query?" The result tells us how much attention to pay to each token. This is a trick to see how aligned 2 vectors are. The more aligned the key to the query the larger the value of this operation. Then we use those attention weights to create a weighted sum of Values, mixing together the content from tokens that seemed relevant.
:::

::: tip 💡 Intuition: Why Scale by sqrt(d)?
When you compute dot products between high-dimensional vectors, the values get very large. If your vectors have dimension `d`, and each element is roughly standard normal, the dot product has variance roughly equal to `d`. When you shove numbers that large into softmax, one position might be ~1.0 and everything else gets ~0.0. This is called "softmax saturation".

Dividing by `sqrt(d)` keeps the variance at roughly 1. It's a small trick that makes a huge difference!
:::

::: tip 💡 Intuition: Why the Causal Mask?

The causal mask ensures each position can only attend to earlier positions (and itself). We fill the upper triangle of the attention matrix with `-inf` because `softmax(-inf) = 0`, so those positions contribute nothing to the weighted sum.

This creates an information flow that only goes left-to-right.
:::

::: tip 💡 Intuition: Why Softmax?
We need to turn our raw attention scores into a probability distribution in such a way that keys that were more aligned to the query contribute more.
:::

## Transformer Block

Let's incorporate this into our previous forward function. First we need a feed-forward network:

```python
def ffn(params, x):
    return jax.nn.relu(x @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]

def transformer_block(params, x):
    x = x + attention(params, x)
    x = x + ffn(params, x)

    return x
```

::: tip 💡 Intuition: Why the Feed-Forward Network?
Attention is great at mixing information *between* tokens, but it's actually just a weighted average. It's a linear operation! (Once you fix the attention weights, output = weights × values = linear transformation.)

The FFN adds nonlinearity through ReLU and lets each token *process* its information independently. Think of attention as "gathering information from neighbors" and FFN as "thinking about what you gathered."

Why expand to `4 * EMBED_DIM` in the middle? More parameters = more expressive power.
:::

Now let's update `forward` to use transformer blocks:

```diff
 def forward(params, inputs):
-    x = embed(params, inputs)
-
-    return x
+    # convert into embedding
+    x = embed(params, inputs)
+    # apply multiple blocks of attention
+    for block_params in params["blocks"]:
+        x = transformer_block(block_params, x)
+    # map to vocabulary
+    return x @ params["W_o"]
```

::: tip 💡 Intuition: Why Residual Connections?
Notice how we write `x = x + attention(params, x)` instead of just `x = attention(params, x)`? That little `+` is doing heavy lifting!

Deep networks suffer from the "vanishing gradient" problem. By the time gradients propagate back through dozens of layers, they've been multiplied by so many small numbers that they basically vanish.

It's like having an express lane on a highway. Even if the attention block produces very small values, the original information in `x` survives and gradients can still flow.

One way to think about is that this lets each layer learn a *modification* to the input rather than a complete transformation.
:::

Before we run this let's make a new function called `init_params` which initializes the params of each transformer block:

```python
NUM_ATTENTION_BLOCKS = 4
def init_params(rand_key):
    params = {}

    rand_key, subkey = jax.random.split(rand_key)
    # mapping of each token to what its vector representation is
    embedding = jax.random.normal(subkey, (VOCABULARY, EMBED_DIM)) * 0.02

    params["token_embedding"] = embedding

    rand_key, subkey = jax.random.split(rand_key)
    # mapping of context_length to what something at a postion looks like
    positional_embedding = jax.random.normal(subkey, (CONTEXT_LENGTH, EMBED_DIM)) * 0.02

    params["positional_embedding"] = positional_embedding

    params["blocks"] = []

    for i in range(NUM_ATTENTION_BLOCKS):
        block_params = {}

        rand_key, subkey = jax.random.split(rand_key)
        W_q = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_k = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_v = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_o = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        block_params["W_q"] = W_q
        block_params["W_k"] = W_k
        block_params["W_v"] = W_v
        block_params["W_o"] = W_o

        rand_key, subkey = jax.random.split(rand_key)
        W1 = jax.random.normal(subkey, (EMBED_DIM, 4 * EMBED_DIM)) * 0.02
        b1 = jax.numpy.zeros((4 * EMBED_DIM,))

        rand_key, subkey = jax.random.split(rand_key)
        W2 = jax.random.normal(subkey, (4 * EMBED_DIM, EMBED_DIM)) * 0.02
        b2 = jax.numpy.zeros((EMBED_DIM,))

        block_params["W1"] = W1
        block_params["b1"] = b1
        block_params["W2"] = W2
        block_params["b2"] = b2

        params["blocks"].append(block_params)

    # some nonesense with jit
    params["blocks"] = tuple(params["blocks"])
    # The final W_o needs to take use from embeddings and map to vocabulary (so that model returns the actual output)
    rand_key, subkey = jax.random.split(rand_key)
    W_o = jax.random.normal(subkey, (EMBED_DIM, VOCABULARY)) * 0.02

    params["W_o"] = W_o

    return params

params = init_params(rand_key)

inputs, outputs = get_batch(train_data, rand_key)

print(f"{inputs.shape=}")
print(f"{outputs.shape=}")

y = forward(params, inputs)

print(f"{y.shape=}")
```

This should print something like:

```
inputs.shape=(32, 128)
outputs.shape=(32, 128)
y.shape=(32, 128, 65)
```

Did you notice the y.shape changed? This is because now we are picking which of the 65 values is most likely! We just made our first prediction.


## Multi-Head Attention

I came here for multi-head attention bro!?!

Ok now that we have single head attention blocks we can start to implement multi-head attention.

In single head attention Q, K, V are `(BATCH, CONTEXT_LENGTH, EMBED_DIM)`. We instead change them to `(BATCH, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIM)`. But why!? The idea is that instead of 1 head learning about the relationships of the tokens in the context in one way, each head can learn how the tokens relate to each other in different ways.

So to achieve this we take `(BATCH, CONTEXT_LENGTH, EMBED_DIM)` and reshape it to `(BATCH, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM)`, then we transpose it to `(BATCH, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIM)`.

That's basically it!

```python

NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
def multihead_attention(params, inputs):
    batch_size = inputs.shape[0]
    context_length = inputs.shape[1]

    Q = inputs @ params["W_q"]
    K = inputs @ params["W_k"]
    V = inputs @ params["W_v"]

    # reshape from (BATCH, CONTEXT_LENGTH, EMBED_DIM) -> (BATCH, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIMS)
    Q = jax.numpy.reshape(Q, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    Q = Q.transpose(0, 2, 1, 3)
    K_new = jax.numpy.reshape(K, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    K_new = K_new.transpose(0, 2, 1, 3)
    V_new = jax.numpy.reshape(V, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    V_new = V_new.transpose(0, 2, 1, 3)

    attention_score = Q @ K_new.transpose(0, 1, 3, 2)

    # scale attention scores otherwise gradients will vanish
    attention_score = attention_score / (HEAD_DIM**0.5)

    # causal mask so that we only look at the previos tokens
    causal_mask = jax.numpy.triu(
        jax.numpy.full((context_length, context_length), -jax.numpy.inf), k=1
    )
    attention_score = attention_score + causal_mask

    attention_weights = jax.nn.softmax(attention_score)

    weighted_sum = attention_weights @ V_new

    # convert back to original shape
    weighted_sum = weighted_sum.transpose(
        0, 2, 1, 3
    )  # (BATCH_SIZE, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIMS)

    weighted_sum = jax.numpy.reshape(
        weighted_sum, (batch_size, context_length, EMBED_DIM)
    )

    return weighted_sum @ params["W_o"]
```

::: tip 💡 Intuition: Why Multiple Heads?
A single attention head can only compute ONE set of attention weights. If it focuses on syntax, it might miss semantics. Multi-head attention allows each head to focus on different types of relationships.

We split the embedding dimension across heads (`HEAD_DIM = EMBED_DIM // NUM_HEADS`), so we're not adding parameters. We're just letting different "subspaces" of the embedding attend differently. Then we concatenate them back together and project through `W_o` to combine what all the heads learned.

In practice, researchers have found that different heads really do learn different things[^5]. Some track syntax, some track coreference, some just look at adjacent tokens. It's beautiful emergent specialization!
:::

Ok now let's update `transformer_block` to use multi-head attention:

```diff
 def transformer_block(params, x):
-    x = x + attention(params, x)
+    attention_out = multihead_attention(params, x)
+    x = x + attention_out
     x = x + ffn(params, x)

     return x
```

Now the output should be:

```
inputs.shape=(32, 128)
outputs.shape=(32, 128)
y.shape=(32, 128, 65)
```

Ok now let's slap a quick training loop on this and see it in action. We'll use Adam[^4] with a learning rate of 3e-4.

```python
def loss_fn(params, inputs, outputs):
    logits = forward(params, inputs)
    return jax.numpy.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, outputs)
    )

TRAINING_STEPS = 5000
def train(data, params, optimizer, optimizer_state, rand_key):
    @jax.jit
    def train_step(params, optimizer_state, inputs, outputs):
        loss, gradients = jax.value_and_grad(loss_fn)(params, inputs, outputs)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss

    for i in range(TRAINING_STEPS):
        rand_key, subkey = jax.random.split(rand_key)
        inputs, outputs = get_batch(data, subkey)

        params, optimizer_state, loss = train_step(
            params, optimizer_state, inputs, outputs
        )

        if i % 100 == 0:
            print(f"loss at {i} == {loss}")

    return params

params = init_params(rand_key)

optimizer = optax.adam(learning_rate=3e-4)
optimizer_state = optimizer.init(params)

params = train(train_data, params, optimizer, optimizer_state, rand_key)
```

If we run this program now we should see something like: 

```
loss at 0 == 4.175050735473633
loss at 100 == 3.1077561378479004
...
```

Woah look at that loss going down, no better feeling in the world!

Now that we have done some training let's test the results and calculate accuracy:

```python
def test(test_data, params, rand_key, num_batches=10):
    total_correct = 0
    total_tokens = 0

    for i in range(num_batches):
        rand_key, subkey = jax.random.split(rand_key)
        inputs, outputs = get_batch(test_data, subkey)

        logits = forward(params, inputs)

        predictions = jax.numpy.argmax(logits, axis=-1)

        total_correct += jax.numpy.sum(predictions == outputs)
        total_tokens += outputs.size

    accuracy = total_correct / total_chars
    return float(accuracy)

accuracy = test(train_data, params, rand_key)

print(f"{accuracy=}")
```

If you run that you should see something like:

```
loss at 4700 == 1.8847413063049316
loss at 4800 == 1.871120810508728
loss at 4900 == 1.8800549507141113
accuracy=0.01656275801360607
```

Ok that accuracy is not that good, I know, but that's not the point of this exercise. However let's see if we can make it better by adding layer normalization to each `transformer_block` as we pass the inputs to `multihead_attention` and the feed forward layer.

## Layer Normalization

First let's add the `layer_norm` function:

```python
def layer_norm(params, x, eps=1e-5):
    mean = jax.numpy.mean(x, axis=-1, keepdims=True)

    variance = jax.numpy.var(x, axis=-1, keepdims=True)

    x_nomalized = (x - mean) / jax.numpy.sqrt(variance + eps)

    return params["gamma"] * x_nomalized + params["beta"]
```

Now let's update `transformer_block` to normalize inputs before attention and FFN:

```diff
 def transformer_block(params, x):
-    attention_out = multihead_attention(params, x)
+    attention_out = multihead_attention(params, layer_norm(params["ln1"], x))
     x = x + attention_out
-    x = x + ffn(params, x)
+    x = x + ffn(params, layer_norm(params["ln2"], x))

     return x
```

::: tip 💡 Intuition: Why Layer Normalization?
In neural networks if values get too big or too small, or if different features have wildly different scales, training becomes unstable. Gradients explode or vanish, and the model throws a tantrum (hehe).

Layer normalization does the following for each token's embedding:
1. Centers everything around zero by subtracting the mean
2. Makes the spread consistent by dividing by the standard deviation
3. Lets the model rescale by using learnable `gamma` and `beta`

Notice we use "pre-norm" style (normalize before attention/FFN) rather than "post-norm" (normalize after). Pre-norm tends to train more stably, especially for deep models[^6].
:::

::: tip 💡 Intuition: Layer vs Batch Normalization?
Why "layer" norm instead of "batch" norm? In language models, normalizing across the batch dimension would mix unrelated sequences. Layer norm normalizes each token independently across its feature dimension, which makes more sense for sequential data.
:::

We also need to update `init_params` to add the layer norm parameters:

```diff
         block_params["W2"] = W2
         block_params["b2"] = b2

+        block_params["ln1"] = {
+            "gamma": jax.numpy.ones((EMBED_DIM,)),
+            "beta": jax.numpy.zeros((EMBED_DIM,)),
+        }
+
+        block_params["ln2"] = {
+            "gamma": jax.numpy.ones((EMBED_DIM,)),
+            "beta": jax.numpy.zeros((EMBED_DIM,)),
+        }
+
         params["blocks"].append(block_params)
```

Now let's see how we do!

```
loss at 4700 == 1.4081562757492065
loss at 4800 == 1.4242630004882812
loss at 4900 == 1.4300929307937622
accuracy=0.020957617089152336
```

Nice! That's better!

## Generation

Ok now we have a workable model (I am using workable veeeery loosely here).

Let's generate some text!

```python

def generate(params, prompt, rand_key):
    for i in range(len(prompt), CONTEXT_LENGTH):
        encoded_prompt = jax.numpy.array(encode(prompt))
        padded = jax.numpy.zeros((CONTEXT_LENGTH,), dtype=jax.numpy.int32)
        inputs = padded.at[0:len(prompt)].set(encoded_prompt)

        logist = forward(params, inputs[None, :])
        predictions = logist[0, len(prompt) - 1]

        rand_key, subkey = jax.random.split(rand_key)
        prediction = jax.random.categorical(subkey, predictions / 0.8)

        decoded_prediction = decode([int(prediction)])
        prompt += decoded_prediction

        print(decoded_prediction, end="", flush=True)

prompt = input("> ")

generate(params, prompt, rand_key)
```

If you run this you should get something like:

```
> to be or not to be
 city;
You had been with by you; feel which he common comediates,
I have not draws instruck my tongue
My tent
```

And that's it folks!

For my next post I will be adding KV caching to this implementation to see if we can speed up inference (the `generate` function).

## References

[^1]: Ain't nothing wrong with using coding agents!
[^2]: Popular ones include BPE (Byte Pair Encoding) used by GPT models, SentencePiece used by LLaMA, and tiktoken by OpenAI. They're way smarter than character-level tokenization but harder to implement from scratch.
[^3]: This formula comes from the legendary "Attention Is All You Need" paper (Vaswani et al., 2017). If you haven't read it, it's surprisingly readable!
[^4]: 3e-4 is a pretty standard learning rate for Adam. Karpathy calls it the "most common learning rate for Adam" and who am I to argue.
[^5]: Check out "What do you learn from context? Probing for sentence structure in contextualized word representations" (Hewitt & Manning, 2019) for some cool visualizations of what different attention heads learn.
[^6]: The original Transformer used post-norm, but GPT-2 and most modern models use pre-norm. See "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) for the gory details.
