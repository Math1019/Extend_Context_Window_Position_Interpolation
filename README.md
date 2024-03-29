# Extending Context Window via Positional Interpolation
A paper presentation of the Extending Context Window of Large Language Models via Positional Interpolation paper

## Overview
There are certain limitations when implementing large language models (LLMs). One main limitation is the pre-defined context window size as many tasks or applications will exceed the context window size. When the context length exceeds the pre-trained settings, traditional transformer models tend to exhibit a significant degradation in performance. 

The purpose of this paper is to introduce the use of position interpolation (PI) as a fine-tuning technique to extend context length for various LLMs over other fine-tuning methods such as extrapolation. The paper covers various experiments testing the viability of PI to determine that PI is successful in extending context length up to 32 times while having reduced complexity and only experiencing a small trade-off with model performance.

---------------------------------------
## Positional Embeddings
Before discussing what is positional interpolation, we need to discuss what are positional embeddings.
<details>
<summary>What are positional embeddings and why are they important?</summary>

- Another way to think about this, what would be the problem if a model perceived sequences as “bags of words” instead?

  <details>
  <summary>Hint Answer</summary>

  The problem is that if we had a bag of words then a Transformer cannot make sense of word ordering. For example, we know that the sentence “Tiffany gives Camille a gift” has a completely different meaning than “Camille gives Tiffany a gift”, but the model cannot tell the difference.

</details>
</details>

<details>
<summary>Full Answer</summary>

Positional embeddings only contain information about a token’s position in a sequence, so there is no semantic or syntactic information included. The positional embedding is added to the input embeddings to provide the model with information about the position of each item in the sequence, which allows the model to make sense of word ordering.

</details>
</details>


![image](https://github.com/Math1019/Extend_Context_Window_Position_Interpolation/assets/111295407/9f2a6d05-37ac-4ad2-a95a-8866e7023e8d)

Many LLMs, such as LLaMA will use positional embedding to maintain sense of word order, but we currently face limits with context size as there is a fixed-length nature to these vectors that puts a limit on the maximum sequence length the model can handle.

----------------------------------------
## Problem:
Many LLMs will have tasks that require long context windows, such as summarizing long docs and doing long conversations. The two ways to extend context window is by either training a LLM from scratch or fine-tuning. 
- When training from scratch, it takes a lot of effort and resources to accomplish as there is an increased amount of complexity as there are more parameters since the context-length increased.
- Extrapolation fine-tuning for increased context length also has obstacles with an increased complexity and severe deterioration in model performance.
  - Extrapolation is training on short context windows to inference on longer context windows, so the problem is that many LLMs use positional embeddings that are not able to extrapolate well, which creates a decline in performance.
    - For example, LLaMA and Falcon both use Rotary position embeddings (RoPE). RoPE is applied by computing positional information through a rotation operation in the embedding space, which creates a rotation matrix. This allows a reduction in positional collisions and an enhanced modeling of long-range dependencies, but RoPE by itself is not good at extrapolation. <br><br>

<details>
  <summary>Why might RoPE not work well through extrapolation?</summary>
  <br>
  <details>
    <summary>Hint:</summary>
    Extrapolation involves training on short context windows to inference on longer context windows. RoPE creates a rotation matrix based on the sequence positions seen during training.
  </details>
  <details>
    <summary>Answer:</summary>
    RoPE has a fixed rotation pattern as its rotation matrices are designed based on the sequence positions seen during training. When encountering longer sequences, the rotation patterns for these new, unseen positions are already outside of the trained context length. This leads to unpredictable or suboptimal embedding rotations. An effective approach needs to ensure that the positional embeddings for longer sequences fall within the trained range.
  </details>
</details>



![image](https://github.com/Math1019/Extend_Context_Window_Position_Interpolation/assets/111295407/cdf46237-6a31-48a7-b451-208606d85680)

This ends up with pre-trained LLM models that use RoPE to have a severe performance issue once it passes the trained context length as it will have high attention scores that will hurt the self-attention mechanism, which is represented in the graphics above. The above graphics show that using extrapolation for RoPE will cause model performance to have higher attention scores, which represent increased model complexity and model performance to decline.

---------------------------------------

## Positional Interpolation (PI):

To get positional embeddings to fall within the training range, this has led to the idea of positional interpolation where we directly **down-scale the position indices** so that the maximum position index matches the previous context window limit. This is done by simply downscaling and dividing the position index by a scaling factor.
<figure>
  <figcaption><i>Extrapolation vs. Interpolation Down-Scale on RoPE</i></figcaption>
  <img src="https://github.com/Math1019/Extend_Context_Window_Position_Interpolation/assets/111295407/c2e79589-94d7-48a2-bd16-5eeb592c27d0" alt="Alternative Text">
</figure>

The top graphic is showing a LLaMa model with a 2048 context window length. The red part of the graphic is when we have gone over the context window length via extrapolation. The bottom graphic shows that in positional interpolation, we downscale the position indices so that we get the 4096 position to still reside in a 2048 context length, which we can see with the increased number of dots in the bottom graphic.


The paper is focused on how to extend the context window when a LLM is using RoPE. Given that RoPE is defined by the $f(x,m)$ below:


$f(x, m) = [(x_0 + ix_1)e^{im\theta_0}, (x_2 + ix_3)e^{im\theta_1}, ..., (x_{d-2} + ix_{d-1})e^{im\theta_{d/2-1}}]^T$

where $i := \sqrt{-1}$ is the imaginary unit and $\theta_j = 10000^{-2j/d}$. 

Using RoPE, the self-attention score $a(m, n)$

$$
= \text{Re}\left(f(q(m), f(k, n))\right)
$$

$$
= \text{Re} \left[ \sum_{j=0}^{d/2-1} (q_{2j} + iq_{2j+1})(k_{2j} - ik_{2j+1})e^{i(m-n)\theta_j} \right]
$$

$$
= \sum_{j=0}^{d/2-1} \left[ (q_{2j}k_{2j} + q_{2j+1}k_{2j+1}) \cos((m - n)\theta_j) + (q_{2j}k_{2j+1} - q_{2j+1}k_{2j}) \sin((m - n)\theta_j) \right]
$$

$$
:= a(m - n)
$$


This gives the self-attention that is only dependent on relative position *m-n*.

Thus to use the positional interpolation in RoPE so that we can scale down each input position index (m) to be within the range [0,L) to fit within the pre-trained context length, we need to change the *f(x,m)* function seen above to:

$$
f'(x,m) = f \left( x, \frac{mL}{L'} \right)
$$

Where:
- x is the word embedding, which is without the position information
- m is the token position/positional embedding
- L is the original context window length/the max length
- L' is the longer extended context window length.

By aligning the ranges of position indices and relative distances before and after extension, the problem with attention score computation due to context window extension is mitigated, so the model is able to easier adapt as the interpolation bound is much tighter than the extrapolation bound for attention scores computed using the interpolated positions.

---------------------------------------

## Pseudocode: Positional Embeddings with Positional Interpolation

- **Input**: 
  - $\ell\in L'$: Position of a token in the sequence.
  - $L$: The original context window length.
  - $L'$: The longer extended context window length.

- **Output**: 
  - $e_p \in \mathbb{R}^{d_e}$: The vector representation of the position with interpolation.

- **Parameters**: 
  - $W_p$: The positional embedding parameter.
  - For $0 \leq i < d_e/2$:

$$ W_{p}(2i - 1, \ell) = \sin\left(\frac{\ell * (L/ L')}{L^{2i/d_{e}}}\right) $$

$$ W_{p}(2i, \ell) = \cos\left(\frac{\ell * (L/ L')}{L^{2i/d_{e}}}\right) $$
 

- **Return**: 
  - Retrieve the positional embedding with interpolation: $e_p = W_p(i, \ell)$ For $0 \leq i < d_e/2$.

The positional embeddings with interpolation are used with the token embeddings to form a token's initial embedding:
- **Variables**:
  - $e$: The token embedding
  - $W_e$: The word embedding matrix
  - $x$: The document
  - $\ell$: Position of a token in the sequence
  - $W_p[:,\ell]$: The Positional embedding with interpolation

$$ e = W_e[:,x[\ell]] + W_p[:,\ell] $$ 

---------------------------------------

## Experiments
Since positional interpolation does not modify the model architecture or attention mechanism, it can be used for a variety of tasks that can help extend the context length. The paper covered many different experiments to evaluate PI effectiveness compare to extrapolation fine-tuned and non-fine-tuned models: 
  - Long Sequence Language Modeling
  - Passerkey Retrieval
  - Benchmarks on Original Context Window Size of 2048
  - Long Document Summarization

Essentially, the paper saw that models fine-tuned with positional interpolation can achieve better perplexity with longer context windows while only seeing a very minor degradation in performance since the experiments showed that PI can help extend the context window up to 32 times successfully.

This can be seen in the below graphic where we see that models fine-tuned with PI shows progressively lower perplexity with longer context window, while the perplexity of fine-tuned with extrapolation increases over the longer window context without even achieving the same context length.


<p align="center">
  <i>Long Sequence Language Modeling Experiment on LLaMA with RoPE</i>
</p>
<p align="center">
  <img src="https://github.com/Math1019/Extend_Context_Window_Position_Interpolation/assets/111295407/268bcb12-5d42-4d93-b5f8-16b8e0e6f859" alt="Alternative Text">
</p>

The example experiments show how positional interpolation can effectively extend a model’s context window to be significantly larger through minimal fine-tuning, and it does not need to modify the model architecture or attention mechanism. The ability to preserve its original architecture gives positional interpolation versatility to be used in various tasks and help models achieve an extended context window.

---------------------------------------
## Critical Analysis 
There are a few things that could have been developed further in the paper.

1. There is potential for extrapolation fine-tuning to use regularization to possibly end up in the [0,L] boundary, but the authors did not conduct a comparison to see how including regularization in extrapolation compares to positional interpolation.
2. The authors discussed how PI can be used in Retrieval-augmented LLMs, few-shots learning, recurrent transformers, and memory transformers, which were all not included in the experiments covered in the paper. Additional work can be done by not only conducting experiments on these models, but the authors can also conduct experiments on tasks that are more sensitive to positional embeddings like question-answering.
3. Experiment how PI performs with different models besides LLaMA or other positional embedding techniques as new discoveries occur. The authors specifically only focus on LLaMA and RoPE when conducting their experiments, but further work can be done to see how PI performs across different models and other embedding techniques.

---------------------------------------
## Additional Resources
- Extending Context is Hard: https://kaiokendev.github.io/context
- Extending the Context length of Language Models: Understanding Positional Interpolation (Blog Post 1): https://medium.com/@jain.sm/extending-the-context-length-of-language-models-a-deep-dive-into-positional-interpolation-a93140c69f6a
- Position Interpolation: Extending Context Window Sizes in Large Language Models (Blog Post 2): https://medium.com/@jain.sm/position-interpolation-extending-context-window-sizes-in-large-language-models-ef19d0209a9f
- Exploring Ways to Extend Context Length in Transformers: https://muhtasham.github.io/blog/posts/explore-context/
- Extending context size via RoPE scaling (with Reddit Discussion Link): https://github.com/ggerganov/llama.cpp/discussions/1965 
---------------------------------------
## Citation:
Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending context window of large language models via positional interpolation. Retrieved from http://arxiv.org/abs/2306.15595
