# FS-EEND

The official Pytorch implementation of "Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors".

This work is submitted to ICASSP 2024.

[Paper :star_struck:](https://arxiv.org/abs/2309.13916) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/FS-EEND/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](liangdi@westlake.edu.cn)

# Introduction

This work proposes a frame-wise online/streaming end-to-end neural diarization (FS-EEND) method in a frame-in-frame-out fashion. To frame-wisely detect a flexible number of speakers and extract/update their corresponding attractors, we propose to leverage a causal speaker embedding encoder and an online non-autoregressive self-attention-based attractor decoder. A look-ahead mechanism is adopted to allow leveraging some future frames for effectively detecting new speakers in real time and adaptively updating speaker attractors. The proposed method processes the audio stream frame by frame, and has a low inference latency caused by the look-ahead frames.

<div align="center">
<image src="/utlis/arch.png"  width="200" alt="The proposed FS-EEND architecture" />
</div>
