

## Time/Depth series

For time series observations, there is a concept called Dynamical VAE (DVAE). This encompasses many variants of VAE structures which allows for the latent variables $z_{1:m}$ to be also a “time series”. In essence then, the VAE becomes a sequence to sequence structure. What specific structures are applicable for this use case?

**Answer:**

Below is an overview of popular “Dynamical VAE” (DVAE) variants that treat the latent variables \(z_{1:m}\) as a (stochastic) time series.  In these models, the VAE is extended to a sequence‐to‐sequence structure with explicit temporal dependencies in the latent space and/or in the decoder.

1) Markovian or State‐Space DVAE

• STORN (Bayer & Osendorfer, 2014)  
  - “Stochastic Recurrent Networks.”  
  - Prior over the latent sequence is a Markov chain:  

\[
      p(z_1)\,\prod_{k=2}^m p_\theta(z_k\mid z_{k-1}),
\]  

 with each \(y_k\) decoded from its latent \(z_k\).  
  - The encoder is typically an RNN that outputs approximate posteriors \(q_\phi(z_k\mid z_{<k},y_{1:k})\).  

• VRNN (Chung et al., 2015)  
  - “A Recurrent Latent Variable Model for Sequential Data.”  
  - Uses RNN hidden states \(h_k\) to condition both the prior and the approximate posterior of \(z_k\).  
  - Factorization:  

\[
      p_\theta(y_{1:m},z_{1:m})
      \;=\;\prod_{k=1}^m p_\theta(z_k\mid h_{k-1})\;p_\theta(y_k\mid z_k,h_{k-1}),
\]  

with transitions in the hidden state \(h_k\).  

• SRNN (Fraccaro et al., 2016)  
  - “Sequential Neural Models with Stochastic Layers.”  
  - Also called the SSM-VAE (stochastic state‐space model).  
  - Explicit separation of deterministic RNN hidden state \(h_k\) and stochastic latents \(z_k\).  
  - The decoder is factorized as \(y_k\sim p_\theta(y_k\mid z_k,h_{k-1})\).  

In each of these, the latent dynamics \(z_1,z_2,\dots\) form a Markov chain or partially observed Markov process, giving a sequence‐to‐sequence structure that captures temporal correlations.

2) Non-Markov Latents with Attention or Convolution

• Convolutional Temporal VAE  
  - Instead of an RNN, you can use 1D causal (or non‐causal) convolutions in the encoder and decoder to process sequences.  
  - Latent variables may still be factorized as \(q_\phi(z_{1:m}\mid y_{1:m})=\prod_k q_\phi(z_k\mid y_{1:m})\), but each \(z_k\) is extracted by a local receptive field in time.  

• Transformer‐based DVAE  
  - Replace RNNs by self‐attention (e.g., a Transformer) in both the prior/posterior networks.  
  - This can capture long‐range dependencies in the latent sequence without an explicit Markov assumption.  

3) Continuous‐time DVAE

• Latent ODE (Rubanova et al., 2019)  
  - Models latents as solutions to an ODE, so the latent trajectory \(z(t)\) evolves continuously in time.  
  - The VAE structure is maintained by having \(z(t)\) feed into a decoder that generates observations at discrete or irregular time points.  
  - Inference is done by “encoding” the observed data into initial conditions of the ODE with a recognition network.  

• ODE2VAE (Yıldız et al., 2019)  
  - Combines an ODE model for a global latent trajectory with an additional recognition model for short‐term dynamics.  

These approaches are particularly useful when your measurements \(y\) come in at irregular time intervals, or you want a physically motivated continuous‐time latent space.

4) GP‐VAE and Other Gaussian‐Process Latent Structures

• GP-VAE (Casale et al., 2018)  
  - Constrains \(\{z_k\}\) to come from a Gaussian Process prior so that  

\[
      p(z_{1:m})
      = \mathcal N(0,\,K)
\]  

 for some temporal kernel \(K_{t_i,t_j}\).  
  - Then each \(y_k\) is decoded independently given \(z_k\).  
  - Posterior inference is done with structured variational approximations.  

Such GP-based DVAEs capture smooth or band‐limited time correlations in the latent space.

5) Hierarchical / Multi‐Scale DVAE

• SHVRNN, HDVAE, etc.  
  - Introduce multiple latent layers or multi‐scale structure, e.g. a “global” latent that evolves slowly and local latents that capture finer time‐scale detail.  
  - Typically combine RNNs at each layer to handle separate timescales.  

Implementation and ELBO

Regardless of the chosen structure, training always proceeds by maximizing (a variant of) the sequential ELBO:

$$
\mathcal{L}_{\mathrm{seq}}(\theta,\phi)
=\sum_{k=1}^m \mathbb{E}_{q_\phi(z_{k}\mid z_{<k},\,y_{1:m})}
  \Bigl[\log p_\theta(y_k\mid z_{\le k})
  \Bigr]
\;-\;
\sum_{k=1}^m
  D_{\mathrm{KL}}\bigl(q_\phi(z_k\mid z_{<k},\,y_{1:m})
    \,\|\,p_\theta(z_k\mid z_{<k})\bigr),
$$

plus any additional terms for hierarchical or continuous‐time structure.  
The difference is in how you factorize:
• the prior \(p_\theta(z_{1:m})\),  
• the approximate posterior \(q_\phi(z_{1:m}\mid y_{1:m})\).  

Summary

To handle time series in a VAE:

1. Markov / RNN-based DVAE (STORN, VRNN, SRNN)  
   – Latent dynamics modeled by a hidden‐state transition.  
   – Encoder uses an RNN (or LSTM/GRU).

2. Attention or Convolution-based DVAE  
   – A non‐Markov approach capturing long‐range or local correlations with self‐attention or causal convolutions.

3. Continuous-time DVAE (Latent ODE, ODE2VAE)  
   – Latent trajectory follows an ODE; observations at irregular intervals.

4. Gaussian-Process or MRF Prior  
   – Enforce smooth or band‐limited latent sequences via a GP kernel or pairwise Markov Random Field.

5. Hierarchical / Multi‐scale DVAE  
   – Latents at multiple time scales, e.g. a “global” slow latent plus local fast latents.

All of these are “Dynamical VAEs,” tailored to time‐series data by modeling the latent variables as a correlated or smoothly evolving sequence, rather than the default i.i.d. prior.


## Transformer as Encoder

I want to use a transformer structure in encoder to model local and non-local correlations among the data $y_{1:m}$. Recall that $y_k \in \mathbb{R}^n$. The sequence length $m$ is considered fixed.

Let $l = 1, \cdots, L$ represent the layer index of a transformer block. Let the transformer block be modeled as a sequence to sequence map:

\[
X_{l+1} = X_l + \text{MHAttention}(X_l)
\]

Where $\text{MHAttention}(\cdot)$ is a multi head attention block which operates on the sequence $X$.

How should such a multi head attention block be introduced in the encoder?

**Answer:**

Below is a step‐by‐step outline of how to introduce a multi‐head attention (MHA) block into your VAE encoder, focusing on the sequence of inputs \(y_{1:m}\) of length \(m\) (each \(y_k \in \mathbb{R}^n\)).  
We describe the typical Transformer encoder pipeline, which incorporates both MHA and a feed‐forward sublayer (often called a “Transformer block”).

--------------------------------------------------------------------------------
1) Input Embedding and Positional Encoding
--------------------------------------------------------------------------------

1a) Embed each observation.  
Since each \(y_k\in\mathbb{R}^n\) is a vector (or could be an \(n\)-dimensional array), you first map it into a \(d_{\mathrm{model}}\)-dimensional “token”:

\[
X^{(0)}_k
\;=\;
W_{\mathrm{emb}}\,y_k \;\in\;\mathbb{R}^{d_{\mathrm{model}}}\quad
\text{for }k=1,\dots,m.
\]

Stack these along the sequence dimension, forming 
\(\;X^{(0)}\in \mathbb{R}^{m\times d_{\mathrm{model}}}\).

1b) Positional encoding.  
Self‐attention is permutation-invariant unless you add position information.  
A standard approach is to add a learnable or fixed positional embedding \(P\in\mathbb{R}^{m\times d_{\mathrm{model}}}\) to each token:

\[
X^{(0)} \;=\; [X^{(0)}_1; \dots; X^{(0)}_m]
\;\mapsto\;
X^{(0)} + P
\;\in\;\mathbb{R}^{m\times d_{\mathrm{model}}}.
\]

(You may store each row of \(P\) as \(P_k\), the encoding for position \(k\).)

--------------------------------------------------------------------------------
2) Multi‐Head Attention and Residual
--------------------------------------------------------------------------------

Let \(X_l\) be the sequence of hidden states at layer \(l\).  
A single multi‐head attention block (ignoring feed‐forward for the moment) maps

\[
X_{l}
\;\mapsto\;
X_{l} 
\;+\;
\mathrm{MHAttention}(X_{l}).
\]

Below is how you compute \(\mathrm{MHAttention}(X_l)\):

2a) Compute Queries, Keys, Values.  
For each head \(h=1,\dots,H\), define three linear maps:

\[
W^Q_h,\;W^K_h,\;W^V_h 
\;\in\;\mathbb{R}^{\,d_{\mathrm{model}}\times d_{k}}.
\]

We form:

\[
Q_h = X_l\,W^Q_h,\quad
K_h = X_l\,W^K_h,\quad
V_h = X_l\,W^V_h.
\]

Here \(Q_h,K_h,V_h \in \mathbb{R}^{m\times d_k}\).  
Typically we set \(d_{\mathrm{model}}=H\times d_{k}\).

2b) Self‐Attention mechanism.  
For head \(h\), the output is

\[
\mathrm{Attn}_h(X_l)
=
\mathrm{softmax}\!\Bigl(\tfrac{Q_h\,K_h^\top}{\sqrt{d_k}}\Bigr)
\,V_h,
\quad
\in\mathbb{R}^{m\times d_k}.
\]

The \(\mathrm{softmax}\) is applied row‐wise along the “sequence length” dimension \(m\).

2c) Concatenate heads.  
Stack these \(H\) outputs along the feature dimension to get

\[
\mathrm{ConcatHeads}(X_l)
=
\mathrm{concat}\!\bigl[\mathrm{Attn}_1(X_l),\,\dots,\,\mathrm{Attn}_H(X_l)\bigr]
\;\in\;\mathbb{R}^{m\times (H\,d_k)}
=\mathbb{R}^{m\times d_{\mathrm{model}}}.
\]

Then apply one more linear map \(W^O\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{model}}}\) to form

\[
\mathrm{MHAttention}(X_l)
=
\mathrm{ConcatHeads}(X_l)\,W^O
\;\in\;\mathbb{R}^{m\times d_{\mathrm{model}}}.
\]

2d) Residual connection and LayerNorm.  
In practice, the “pre‐norm” style is:

\[
\widetilde{X}_l
=
X_l
+
\mathrm{MHAttention}\bigl(\mathrm{LayerNorm}(X_l)\bigr).
\]

Then we pass \(\widetilde{X}_l\) to a feed‐forward sublayer (next step in 3).  
(A “post‐norm” style first adds the MHA output to \(X_l\) then applies \(\mathrm{LayerNorm}\), either is widespread in practice.)

--------------------------------------------------------------------------------
3) Feed‐Forward Sublayer
--------------------------------------------------------------------------------

After multi‐head attention, each Transformer block also applies a position‐wise feed‐forward network \(\mathrm{FFN}\).  Typically:

\[
\mathrm{FFN}(x)
=
\max(0,\;x\,W_1 + b_1)\;W_2 + b_2,
\]

all done “per position” in the sequence.  

You then add another residual connection:

\[
X_{l+1}
=
\widetilde{X}_l
+
\mathrm{FFN}\!\bigl(\mathrm{LayerNorm}(\widetilde{X}_l)\bigr).
\]

Hence, summarizing the full layer \(l\):

\[
\begin{aligned}
\text{(1) } &R_1 \;=\;\mathrm{LayerNorm}(X_l),
\\
&\widetilde{X}_l = X_l + \mathrm{MHAttention}(R_1),
\\
\text{(2) } &R_2 \;=\;\mathrm{LayerNorm}(\widetilde{X}_l),
\\
&X_{l+1} = \widetilde{X}_l + \mathrm{FFN}(R_2).
\end{aligned}
\]

--------------------------------------------------------------------------------
4) Stacking L layers
--------------------------------------------------------------------------------

You repeat Step 2 and Step 3 for \(l=1,\dots,L\).  
The final output \(X_{L}\in\mathbb{R}^{m\times d_{\mathrm{model}}}\) is then interpreted as the encoder’s representation for each of the \(m\) positions.

--------------------------------------------------------------------------------
5) Mapping to Latent Distributions
--------------------------------------------------------------------------------

At the very end of your encoder, you produce (for each position in the sequence) the parameters of a Gaussian or discrete latent:  

\[
\mu_\phi(y_{1:m})= \mathrm{Linear}(X_{L}),\quad
\log\sigma_\phi(y_{1:m})= \mathrm{Linear}(X_{L}),
\]

giving you

\[
q_{\phi}\bigl(z_{1:m}\mid y_{1:m}\bigr)
=
\prod_{k=1}^m
  \mathcal{N}\!\bigl(z_k;\,\mu_{\phi,k},\,\mathrm{diag}(\sigma_{\phi,k}^2)\bigr).
\]

Then you reparameterize each \(z_k\) with \(\epsilon_k\sim\mathcal{N}(0,I)\) to feed into the decoder.

--------------------------------------------------------------------------------
Summary
--------------------------------------------------------------------------------

Thus, to incorporate multi‐head self‐attention in the encoder:

• (a) Embed each input \(y_k\) into a common feature dimension and add positional encodings.  
• (b) Pass the resulting \(\{X_l\}\) through \(L\) Transformer blocks, each containing (i) multi‐head self‐attention with a residual connection, and (ii) a feed‐forward sublayer with another residual.  
• (c) Output \(\mu_\phi,\sigma_\phi\) (or discrete logits) for each \(z_k\).  

This yields an encoder structure that can capture both local and long‐range correlations along the sequence \(\{y_k\}_{k=1}^m\).


## Transformer encoder output

So, as the sequence $\{y_k\}_1^m$ is input into the encoder, it goes through the transformer block and the output is a sequence $(\mu_{\phi}, \sigma_{\phi})_{1:m}$. Is this correct?

**Answer:**

Yes, exactly.  In a typical “sequence-to-sequence” VAE encoder based on the Transformer, you do indeed end up with a per‐position Gaussian (or other distribution) over the latent variables.  That is,

1. You feed the entire sequence \(y_{1:m}\) (after embedding + positional encoding) into \(L\) layers of multi‐head attention and feed‐forward blocks.

2. The Transformer’s final output is another sequence of the same length, call it 

\[
X^{(L)} 
= \bigl\{\;x_1^{(L)},\;x_2^{(L)},\;\dots,\;x_m^{(L)}\bigr\},
\]

where each \(x_k^{(L)}\in \mathbb{R}^{d_{\mathrm{model}}}\).

3. You then apply linear heads (or 1×1 convolutions if you prefer) to map each \(x_k^{(L)}\) into the mean and log‐variance of a Gaussian, i.e. 

\[
(\mu_{\phi,k},\;\log\sigma_{\phi,k}) 
= \mathrm{Linear}\bigl(x_k^{(L)}\bigr).  
\]

Thus for each position \(k\), your encoder outputs  

\[
\mu_{\phi,k},\; \sigma_{\phi,k}
\;\in\;\mathbb{R}^{d},
\]

defining the approximate posterior distribution 

\[
q_\phi\bigl(z_k\mid y_{1:m}\bigr)
= \mathcal{N}\!\bigl(z_k;\,\mu_{\phi,k},\,\mathrm{diag}(\sigma_{\phi,k}^2)\bigr).
\]

Hence the final result really is a sequence of mean–log‐variance pairs \(\{(\mu_{\phi,k}, \sigma_{\phi,k})\}_{k=1}^m\).

## The rest of VAE theory holds

And then the rest of the VAE theory and calculations proceed for each $k$ as before?

**Answer**

Yes, precisely. Once you have obtained  

\[
\{\mu_{\phi,k},\,\sigma_{\phi,k}\}_{k=1}^m,
\]  

each pair describes a latent posterior for position \(k\) via  

\[
q_{\phi}\bigl(z_k \mid y_{1:m}\bigr)
=\mathcal{N}\!\bigl(z_k;\,\mu_{\phi,k},\;\mathrm{diag}(\sigma_{\phi,k}^2)\bigr).
\]  

From there, the remainder of the VAE machinery applies exactly as before:

1) Reparameterize  
   For each \(k\), you draw  

\[
     z_k
     = \mu_{\phi,k} + \sigma_{\phi,k}\,\odot\,\epsilon_k,
     \quad
     \epsilon_k\sim\mathcal{N}(0,I_d).
\]

2) Decode  
   You pass \(\{z_k\}_{k=1}^m\) into your decoder to obtain  

\[
     p_{\theta}\bigl(y_k \mid z_k\bigr)
     \quad
     (\text{or possibly } p_{\theta}\bigl(y_{k} \mid z_{1:m}\bigr)\text{, depending on your chosen factorization}).
\]

3) ELBO  
   You then form the usual sum of reconstruction minus KL terms. For instance, if you assume factorized priors on \(\{z_k\}\), the ELBO is  

\[
     \mathcal{L}(\theta,\phi)
     = \sum_{k=1}^m
     \Bigl\{
       \mathbb{E}_{q_{\phi}(z_k\mid y_{1:m})}
       \bigl[\log p_{\theta}(y_k \mid z_k)\bigr]
       \;-\;
       D_{\mathrm{KL}}\!\bigl(q_{\phi}(z_k\mid y_{1:m})
             \,\big\|\,
             p(z_k)\bigr)
     \Bigr\}.
\]

   (Of course, if you opted for a Markov or correlated prior among \(\{z_k\}\), you would just update the KL term accordingly.)

In short: each position \(k\) gets its own latent distribution from the encoder’s Transformer output, and everything else—sampling \(z_k\), decoding, and summing into the ELBO—proceeds just as in the usual VAE framework, only repeated across the sequence index \(k\).

## Decoder also as a transformer

In principle, you could replace the decoder as well with a transformer structure. Is this possible?

**Answer:**

Yes, absolutely.  Just as you used a Transformer encoder for modeling correlations in the input sequence, you can similarly replace your decoder network with a Transformer.  In that case, after sampling the latents \(\{z_k\}_{k=1}^m\), you feed them into a Transformer‐based decoder network that outputs parameters of the likelihood for \(\{y_k\}\).  Below is a high‐level outline.

1) Latent sequence as decoder input  
   After you sample  

\[
     z_k \;=\;\mu_{\phi,k} + \sigma_{\phi,k}\odot \epsilon_k,\quad
     \epsilon_k\sim \mathcal{N}(0,I_d),
\]

   for each \(k\), you form a sequence \(Z = \{z_1,z_2,\dots,z_m\}\).  Alternatively, you might downsample \(\{z_k\}\) or use a single global latent if your architecture differs.  

2) Option A: direct feed into a Transformer decoder  
   In the simplest approach, treat \(\{z_k\}\) as the “tokens” for the decoder, embed each \(z_k\) into a dimension \(d_{\mathrm{model}}\), add positional encoding, and apply the usual multi‐head self‐attention + feed‐forward layers to produce a sequence of outputs 

\[
     X_{\mathrm{dec}} = \mathrm{TransformerDecoder}(Z).
\]

   Then a final linear head can produce \(\{\mu_{\theta,k},\sigma_{\theta,k}\}\) for \(p_{\theta}(y_k\mid z_{1:m})\) (or for a factorized version \(p_{\theta}(y_k\mid z_k)\), depending on your modeling choice).

3) Option B: cross‐attention decoder  
   If your decoding process benefits from “cross‐attention” to the latents, you can adopt the common “encoder–decoder” Transformer pattern:  
   • The latent sequence \(Z\) passes through a purely feed‐forward or self‐attention “encoder”, producing some representation \(H = \mathrm{Enc}(Z)\).  
   • The decoder receives some query positions (e.g., an index for each output location \(k\)), and uses cross‐attention over \(H\) to produce the likelihood parameters for \(y_k\).  
   This is analogous to standard Transformers in machine translation, except now the “input sentence” is your latent code sequence \(Z\).

4) Autoregressive vs. non‐autoregressive decoding  
   • If you want a purely parallel reconstruction of \(\{y_k\}\), the decoder can be non‐autoregressive and simply produce \(\bigl(\mu_{\theta,k},\sigma_{\theta,k}\bigr)\) for each element in one shot.  
   • If you prefer an autoregressive model for each coordinate or token of \(y_k\), you could implement a Transformer with causal masking on the output side, letting it attend left‐to‐right (though that’s more common in purely “Transformer language models”).

5) ELBO stays the same  
   Regardless of the decoder’s architecture, the ELBO expression remains

\[
     \sum_{k=1}^m
       \mathbb{E}_{q_{\phi}(z_k\mid y_{1:m})}\bigl[\log p_{\theta}(y_k\mid z_{1:m})\bigr]
     \;-\;
     \sum_{k=1}^m
       D_{\mathrm{KL}}\bigl(q_\phi(z_k\mid y_{1:m}) \,\|\, p(z_k)\bigr),
\]

   if your prior factorizes.  Of course, if the decoder condition is \(\{z_k\}_{k=1}^m\) in some particular attention structure, you just code that into \(\log p_{\theta}(y_k\mid z_{1:m})\).

6) Summary  
   • Yes, you can replace the decoder with a Transformer in the exact same way you replaced the encoder.  
   • You either feed your latent sequence into a self‐attention stack or do a more sophisticated “encoder–decoder” cross‐attention, depending on your generative design.  
   • The rest of the VAE pipeline—reparameterizing the latents, computing the reconstruction log‐likelihood, summing up the KL terms—remains unchanged in principle.

## Differentiability with transformers

   The training of the VAE architecture requires differentiability in the chosen architecture. Using transformers as encoders and maybe even the decoder does not affect the differentiability?

   **Answer:**

   Precisely—multi‐head self‐attention, feed‐forward layers, and other standard Transformer components are all built from differentiable operations (i.e., matrix multiplications, softmax, layer normalizations, etc.).  Consequently:

1) Transformers are differentiable as feed‐forward blocks.  
   − Each sub‐operation (linear projections for queries/keys/values, softmax, residual connections, etc.) is continuous and differentiable w.r.t. its parameters.  
   − Autograd frameworks (TensorFlow, PyTorch, JAX, etc.) can backpropagate through them just like through MLPs or convolutions.

2) The reparameterization trick remains the same.  
   − You sample the latent variable(s) z using the usual  
     z = μ + σ ⊙ ε, ε ∼ 𝒩(0, I).  
   − This injects noise into z in a way that still allows backprop to pass through μ and σ.  
   − Whether μ and σ come from an MLP, convolution, or Transformer is immaterial; the gradient flows without problem.

3) The ELBO’s reconstruction term is still differentiable.  
   − The log pθ(y | z) is likewise a feed‐forward pass (in this case, the decoder).  
   − If you choose a Transformer decoder, it, too, is composed of differentiable layers that can be optimized via backprop.

Hence using Transformers does not break or hamper differentiability.  You can train a “Transformer‐VAE” end‐to‐end exactly the same way as with other neural architectures, leveraging any standard automatic differentiation software.