---
layout: doc
---

# Tiny LLM go brrrrr

*March 31, 2026*

---

In part [2/2] we will cover KV caching.

::: info 📝 Note
I build on part [1/2](/blog/tinyLLM) here, so I highly recommend giving that a read before starting this one.
:::

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

We implemented multi-head attention in part [1/2](/blog/tinyLLM) and used that to generate text.

When measuring the performance of generation the key metrics are[^1]:

1. Time to first token (TTFT)
2. Time per output token (TPOT)
3. Inter-token latency (ITL)
4. End-to-end latency (E2EL)

Let's instrument these in our `generate` function.

```python
from datetime import datetime

def generate(params, prompt, rand_key):
    token_times = []
    for i in range(len(prompt), CONTEXT_LENGTH):
        token_start = datetime.now()

        encoded_prompt = jax.numpy.array(encode(prompt))
        padded = jax.numpy.zeros((CONTEXT_LENGTH,), dtype=jax.numpy.int32)
        inputs = padded.at[0:len(prompt)].set(encoded_prompt)

        logist = forward(params, inputs[None, :])
        predictions = logist[0, len(prompt) - 1]

        rand_key, subkey = jax.random.split(rand_key)
        prediction = jax.random.categorical(subkey, predictions / 0.8)

        decoded_prediction = decode([int(prediction)])
        prompt += decoded_prediction

        token_end = datetime.now()
        token_times.append((token_end - token_start).total_seconds() * 1000)

        print(decoded_prediction, end="", flush=True)

    end_time = datetime.now()
    print()

    ttft = token_times[0]  # Time to First Token (ms)
    tpot = sum(token_times[1:]) / (len(token_times) -1)  # Time Per Output Token (ms)
    avg_itl = tpot  # Inter-Token Latencies is bacially tpot for a single request

    print(f"TTFT: {ttft:.2f}ms")
    print(f"TPOT: {tpot:.2f}ms")
    print(f"Avg ITL: {avg_itl:.2f}ms")
    print(f"E2ET: {sum(token_times):.2f}ms")
```

If you run this you should get something like:

```
> to be or not to be
 city;
You had been with by you; feel which he common comediates,
I have not draws instruck my tongue
My tent 

TTFT: 994.40ms
TPOT: 53.94ms
Avg ITL: 53.94ms
```

Before we dive into KV Cache let's add `@jax.jit`[^2] to `forward` and see if that gives us some performance gains.

```
> to be or not to be
 city;
You had been with by you; feel which he common comediates,
I have not draws instruck my tongue
My tent 

TTFT: 511.33ms
TPOT: 51.47ms
Avg ITL: 51.47ms
E2EL: 6121.61ms
```

Ok that was neat! But can we do better?

## KV Cache

Let's think about what happens during generation. When we generate the 10th token, we run the full forward pass on all 10 tokens. Then when we generate the 11th token, we run the forward pass on all 11 tokens. See the problem? We're recomputing attention for the previous tokens each time.

In attention, for each token we compute:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "Here's my content"

The K and V for tokens 1-10 are exactly the same when we generate token 11. We're wasting compute recalculating them. KV caching solves this by storing the K and V values from previous tokens and reusing them.

::: tip 💡 Intuition: Why Cache K and V but not Q?
During generation, we only need the Query for the *new* token. The new token asks "what should I attend to?" and we look up the answer using the cached Keys and Values from all previous tokens.
:::

For KV caching we are going to break generation into 2 parts:
1. **Prefill**: The initial values of the KV caches and the first token are going to be generated in this phase. We should not see an improvement here.
2. **Decode**: The new tokens are going to be generated in this phase. We will only send the last token instead of the whole `prompt + last token` like we were doing before. This phase should see a speedup and the TPOT should improve because we will use the caches from prefill.

::: tip 💡 Intuition: Multi-head means multiple Ks and Vs are cached
We have multiple attention blocks, so we will need a KV cache per block.
:::

Let's start by updating `multihead_attention` to also return K and V:

```diff
     weighted_sum = jax.numpy.reshape(
         weighted_sum, (batch_size, context_length, EMBED_DIM)
     )

-    return weighted_sum @ params["W_o"]
+    return (weighted_sum @ params["W_o"], K, V)
```

And update `transformer_block` to pass them through:

```diff
 def transformer_block(params, x):
-    attention_out = multihead_attention(params, layer_norm(params["ln1"], x))
-    x = x + attention_out
+    attention, K, V = multihead_attention(params, layer_norm(params["ln1"], x))
+    x = x + attention
     x = x + ffn(params, layer_norm(params["ln2"], x))

-    return x
+    return (x, K, V)
```

Now let's write a new `forward_prefill` function. This function will build the caches and return them along with the first token.

```python
def forward_prefill(params, inputs):
    x = embed_prefill(params, inputs)

    batch_size = inputs.shape[0]
    prompt_length = inputs.shape[1]

    kvs = []
    for block_params in params["blocks"]:
        x, K, V = transformer_block(block_params, x)
        k_cache = jax.numpy.zeros((batch_size, CONTEXT_LENGTH, EMBED_DIM))
        v_cache = jax.numpy.zeros((batch_size, CONTEXT_LENGTH, EMBED_DIM))

        k_cache = k_cache.at[:, :prompt_length].set(K)
        v_cache = v_cache.at[:, :prompt_length].set(V)

        kvs.append((k_cache, v_cache))

    return (x @ params["W_o"], kvs)
```

For `forward_prefill` we will need to add a new embedding function. This is because we won't pad the input to be `CONTEXT_LENGTH` anymore, so our original `embed` function won't work.

```python
def embed_prefill(params, inputs):
    # we only generate positional embeddings till the prompt length, we don't pad to CONTEXT_LENGTH
    return (
        params["token_embedding"][inputs]
        + params["positional_embedding"][: inputs.shape[1]]
    )
```

Now let's write another function called `forward_decode` that will use these caches and generate the remaining tokens.

```python
@jax.jit
def forward_decode(params, inputs, position, kvs):
    # simulate actual position
    x = embed_at(params, inputs, position)

    new_kvs = []
    for i, block_params in enumerate(params["blocks"]):
        x, k_cache, v_cache = transformer_block_decode(block_params, x, position, kvs[i][0], kvs[i][1])
        new_kvs.append((k_cache, v_cache))

    return (x @ params["W_o"], new_kvs)
```

Since we are only sending 1 token we have to keep track of the position. We'll add another embedding function and a new `transformer_block_decode` function that takes the previous K and V values as input.

```python
def embed_at(params, inputs, position):
    return params["token_embedding"][inputs] + params["positional_embedding"][position]


def transformer_block_decode(params, x, position, pre_K, pre_V):
    attention, K, V = multihead_attention_cached(
        params, layer_norm(params["ln1"], x), position, pre_K, pre_V
    )
    x = x + attention
    x = x + ffn(params, layer_norm(params["ln2"], x))

    return x, K, V
```

We will also write a new `multihead_attention_cached` function which will take the previous K and V values as input.

```python
# inputs is always just (1, 1, EMBED_DIM)
# we have a single batch with just one new token
# k_cache, v_cache are (1, CONTEXT_LENGTH, EMBED_DIM)
def multihead_attention_cached(params, inputs, position, k_cache, v_cache):
    batch_size = inputs.shape[0]
    context_length = inputs.shape[1]
 
    Q = inputs @ params["W_q"]
    K_new = inputs @ params["W_k"]
    V_new = inputs @ params["W_v"]

    k_cache = k_cache.at[:, position].set(K_new[:, 0])
    v_cache = v_cache.at[:, position].set(V_new[:, 0])

    Q = jax.numpy.reshape(Q, (batch_size, 1, NUM_HEADS, HEAD_DIM))
    Q = Q.transpose(0, 2, 1, 3)
    K = jax.numpy.reshape(k_cache, (batch_size, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM))
    K = K.transpose(0, 2, 1, 3)
    V = jax.numpy.reshape(v_cache, (batch_size, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM))
    V = V.transpose(0, 2, 1, 3)

    # attention scores

    attention_score = Q @ K.transpose(0, 1, 3, 2)

    # scale attention scores otherwise gradients will vanish

    attention_score = attention_score / (HEAD_DIM**0.5)

    # causal mask so that we only look at the previos tokens
    causal_mask = jax.numpy.where(
        jax.numpy.arange(CONTEXT_LENGTH) > position, 
        -jax.numpy.inf,
        0.0
    )
    attention_score = attention_score + causal_mask

    attention_weights = jax.nn.softmax(attention_score)

    weighted_sum = attention_weights @ V

    # convert back to original shape
    weighted_sum = weighted_sum.transpose(
        0, 2, 1, 3
    )  # (BATCH_SIZE, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIMS)

    weighted_sum = jax.numpy.reshape(
        weighted_sum, (batch_size, context_length, EMBED_DIM)
    )

    return (weighted_sum @ params["W_o"], k_cache, v_cache)
```

Ok now let's update the `generate` function!

```python
def generate(params, prompt, rand_key):
    token_times = []

    token_start = datetime.now()
    inputs = encode(prompt)

    # note: the [None, :] is to add a batch dimension
    logits, kvs = forward_prefill(params, jax.numpy.array(inputs)[None, :])

    # 1st batch last element
    predictions = logits[0, -1]

    rand_key, subkey = jax.random.split(rand_key)

    prediction = jax.random.categorical(subkey, predictions / 0.8)

    token_end = datetime.now()
    token_times.append((token_end-token_start).total_seconds() * 1000)


    # Print the first generated token
    # This is measure of time to first token!
    print(decode([int(prediction)]), end="", flush=True)

    for i in range(len(prompt), CONTEXT_LENGTH):

        token_start = datetime.now()

        next_token = jax.numpy.array([int(prediction)])
        logits, kvs = forward_decode(
            params, next_token[None, :], i, kvs
        )  # this add the extra batch dimension

        predictions = logits[0, 0]

        rand_key, subkey = jax.random.split(rand_key)

        prediction = jax.random.categorical(subkey, predictions / 0.8)

        token_end = datetime.now()
        token_times.append((token_end-token_start).total_seconds() * 1000)

        # Print each new token as it's generated
        print(decode([int(prediction)]), end="", flush=True)

    # Print newline at the end
    print()

    ttft = token_times[0]  # Time to First Token (ms)
    tpot = sum(token_times[1:]) / (len(token_times) -1)  # Time Per Output Token (ms)
    avg_itl = tpot  # Inter-Token Latencies is bacially tpot for a single request

    print(f"TTFT: {ttft:.2f}ms")
    print(f"TPOT: {tpot:.2f}ms")
    print(f"Avg ITL: {avg_itl:.2f}ms")
    print(f"E2EL: {sum(token_times):.2f}ms")
```

Running the program now should output the following:

```
> to be or not to be
 city;
You had been with by you; feel which he common comediates,
I have not draws instruck my tongue
My tent 

TTFT: 522.87ms
TPOT: 3.94ms
Avg ITL: 3.94ms
E2EL: 956.16ms
```

Look at that TPOT drop! We went from ~50ms to ~4ms per token. That's a 12x speedup on the decode phase. The TTFT stays roughly the same because we still need to process the full prompt on the first pass (prefill).

::: tip 💡 Intuition: Prefill vs Decode characteristics
The two phases of generation have different characteristics:

**Prefill** processes the entire prompt at once. This is compute-bound because we're doing a lot of matrix multiplications. KV cache doesn't help here since we're computing K and V for the first time.

**Decode** generates one token at a time. This is memory-bound because we're mostly just reading the cached K and V values. KV cache shines here since we only compute K and V for one new token.

This is why production LLM systems often report TTFT and TPOT separately. They have very different optimization strategies!
:::

KV caching isn't free. We're trading compute for memory. For each layer, we store:
- K cache: `(BATCH_SIZE, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIM)`
- V cache: `(BATCH_SIZE, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIM)`

With our tiny model that's not much, but for a real LLM, the KV cache can be several gigabytes per request.

If you liked this post please share it with your friends!

You can find the complete implementation [here](https://github.com/vinayakankugoyal/tinyLLM.git).

::: info 📝 Note
You must be wondering why I pre-allocate `k_cache` and `v_cache` to be `CONTEXT_LENGTH`. It's because that makes the size of all inputs the same across calls. This is a common `jax.jit` gotcha where JAX recompiles every time the function is called with different input shapes. If we did not have a constant size for the caches and concatenated the values to the previous K and V values, then every call to `multihead_attention_cached` would cause recompilation and that would massively slow down the program.

I managed to do exactly the above. See how slow the program is by checking out [this commit](https://github.com/vinayakankugoyal/tinyLLM/commit/f300bb15562ba55d71c45991ce9ac8f58b953a5f).
:::

## References

[^1]: These metrics are commonly used in production LLM systems. See the vLLM paper for more details on how they're measured and optimized.
[^2]: `@jax.jit` compiles your function using XLA (Accelerated Linear Algebra) for faster execution. It traces your function once and then runs the optimized version on subsequent calls.
