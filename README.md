# About this repo

This repo contains some of the fun papers in deep learning and other related fields I've come across. I have also implemented some of them myself via NumPy/PyTorch in my spare time.

# Related papers (by categories):

## Computer vision:

### Image restoration:
- **AutoEncoder**: classical :)
- (NeurIPS 2016) **RED-Net**: Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections [[link]](https://proceedings.neurips.cc/paper/2016/hash/0ed9422357395a0d4879191c66f4faa2-Abstract.html)

### Image quality assessment:
- (2004) **SSIM**: Image quality assessment: from error visibility to structural similarity [[link]](https://ieeexplore.ieee.org/document/1284395) [[pdf]](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)

## Natural language processing:

### Text to speech (TTS):
- (2016) **WaveNet**: A Generative Model for Raw Audio [[pdf]](https://arxiv.org/pdf/1609.03499.pdf) [[blog]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

### Decoding strategy:
- (ICLR 2020) **Nucleus Sampling**: The Curious Case of Neural Text Degeneration [[pdf]](https://arxiv.org/pdf/1904.09751) [[link]](https://openreview.net/forum?id=rygGQyrFvH)

## Information theory:

### Minimum description length:

*also known as bits-back coding* 
- (1993) Keeping neural networks simple by minimizing the description
length of the weights [[pdf]](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)
- (ICLR 2019) **BB-ANS**: Practical lossless compression with latent variables using bits back coding [[link]](https://openreview.net/forum?id=ryE98iR5tm) [[pdf]](https://openreview.net/pdf?id=ryE98iR5tm)
  
  *see section 2 for a high-level understanding of bits-back coding*

## Generative model:

### Explicit density:
- (ICLR 2014) **VAE**: Auto-Encoding Variational Bayes [[link]](https://openreview.net/forum?id=33X9fd2-9FyZd) [[pdf]](https://arxiv.org/abs/1312.6114)
- (ICML 2014) **DARN**: Deep AutoRegressive Networks [[link]](https://openreview.net/forum?id=rkZC3qZdZr) [[pdf]](http://proceedings.mlr.press/v32/gregor14.pdf)
- (ICML 2016) **PixelRNN**: Pixel Recurrent Neural Networks [[link]](http://proceedings.mlr.press/v48/oord16.html) [[pdf]](http://proceedings.mlr.press/v48/oord16.pdf)