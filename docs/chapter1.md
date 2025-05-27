

## VAE Architecture 1

Let $y_k$ be measurements of a data generating process. I would like to use the Variational Autoencoder method of statistical inference, to learn about the data generating process. Describe the details of the VAE to accomplish this.

**Answer**

Below is a self-contained description of how to set up and train a Variational Autoencoder (VAE) to infer the data-generating process behind measurements \(y_k\).  All formulas are in standard LaTeX.

1. Model specification  
   We introduce a low-dimensional latent variable \(z\in\mathbb R^d\).  The generative model (decoder) and the inference model (encoder) are

\[
     \text{Prior:}\quad p(z)=\mathcal N\bigl(z;0,I\bigr),
     \quad
     \text{Likelihood:}\quad p_\theta(y\mid z),
\]

\[
     \text{Approx.\ posterior:}\quad
     q_\phi(z\mid y)
     =\mathcal N\bigl(z;\,\mu_\phi(y),\,\mathrm{diag}(\sigma_\phi^2(y))\bigr).
\]

   Here \(\theta\) are the parameters of the decoder network (which outputs parameters of \(p_\theta(y\mid z)\)), and \(\phi\) are the parameters of the encoder network (which outputs \(\mu_\phi(y)\) and \(\sigma_\phi(y)\)).

2. Evidence Lower BOund (ELBO)  
   For a single datum \(y\), the marginal log-likelihood \(\log p_\theta(y)\) is intractable.  We instead maximize the ELBO

\[
     \mathcal L(\theta,\phi\,;\,y)
     \;=\;
     \mathbb{E}_{q_\phi(z\mid y)}\bigl[\log p_\theta(y\mid z)\bigr]
     \;-\;
     D_{\mathrm{KL}}\bigl(q_\phi(z\mid y)\;\|\;p(z)\bigr),
\]

   which satisfies \(\log p_\theta(y)\ge\mathcal L(\theta,\phi;y)\).  Over a dataset \(\{y_k\}_{k=1}^N\) we maximize

\[
     \mathcal L_{\rm total}(\theta,\phi)
     = \sum_{k=1}^N \mathcal L(\theta,\phi\,;\,y_k).
\]

3. Reparameterization trick  
   To backpropagate through the stochastic sampling \(z\sim q_\phi(z\mid y)\), we write

\[
     z \;=\;\mu_\phi(y)\;+\;\sigma_\phi(y)\odot\epsilon,
     \quad
     \epsilon\sim\mathcal N(0,I).
\]

   Then

\[
     \mathbb{E}_{q_\phi(z\mid y)}\bigl[\log p_\theta(y\mid z)\bigr]
     \;=\;
     \mathbb{E}_{\epsilon\sim\mathcal N(0,I)}
     \bigl[\log p_\theta\bigl(y\mid \mu_\phi(y)+\sigma_\phi(y)\odot\epsilon\bigr)\bigr],
\]

   which can be unbiasedly estimated with Monte Carlo and differentiated w.r.t.\ \(\phi\).

4. KL-divergence term  
   For two Gaussians \(q=\mathcal N(\mu,\mathrm{diag}(\sigma^2))\) and \(p=\mathcal N(0,I)\),

\[
     D_{\rm KL}\bigl(q\|p\bigr)
     =\frac12\sum_{i=1}^d
       \bigl[
         \sigma_i^2+\mu_i^2-1-\log(\sigma_i^2)
       \bigr].
\]

5. Overall training algorithm  
   Use mini-batches \(\{y^{(b)}\}\).  For each batch:
   a) Encode: compute \(\mu_\phi(y^{(b)})\), \(\sigma_\phi(y^{(b)})\).  
   b) Sample \(\epsilon\sim\mathcal N(0,I)\), form \(z=\mu_\phi+\sigma_\phi\odot\epsilon\).  
   c) Decode: compute \(\log p_\theta(y^{(b)}\mid z)\).  
   d) Compute batch ELBO 

\[
        \mathcal L_{\rm batch}
         =\frac1B\sum_{i=1}^B
          \Bigl[\log p_\theta(y^{(i)}\!\mid z^{(i)})
            -D_{\rm KL}(q_\phi(z^{(i)}\mid y^{(i)})\|p(z))\Bigr].
\]

   e) Take gradient \(\nabla_{\theta,\phi}\mathcal L_{\rm batch}\) (e.g.\ via Adam) and update \(\theta,\phi\).

6. Choice of decoders  
   – For real-valued \(y\): use Gaussian likelihood  
     \(\;p_\theta(y\mid z)=\mathcal N(y;\,\mu_\theta(z),\,\mathrm{diag}(\sigma_y^2))\).  
   – For binary \(y\): use Bernoulli  
     \(\;p_\theta(y\mid z)=\mathrm{Bernoulli}(y;\,\pi_\theta(z)).\)

7. Inference about the data-generating process  
   After training, the decoder \(p_\theta(y\mid z)\) and the aggregated posterior give you a generative model of the data.  You can  
   • Sample new synthetic observations \(y\sim p_\theta(y\mid z)\) by first drawing \(z\sim p(z)\).  
   • Approximate the latent representation of any \(y\) by \(q_\phi(z\mid y)\).  
   • Inspect learned features by varying individual components of \(z\) and decoding.

This completes the detailed VAE setup for learning the underlying data-generating process from measurements \(y_k\).