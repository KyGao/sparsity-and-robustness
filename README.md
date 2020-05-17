# pruning-recapitulation
merging some articles about network pruning 



已经有了很多篇结合的文章了



- 2020~ [Adversarial Robustness vs Model Compression, or Both?](https://arxiv.org/abs/1903.12561)

这篇文章使用ADMM交叉乘子法进行两步剪枝，第一步正常样本训练，第二步对抗样本训练，达到对抗同时剪枝的目的。类似快手的ATMC

- 2020~ [Adversarial Neural Pruning Latent Vulnerability Suppression](https://arxiv.org/pdf/1908.04355.pdf)

这篇文章走的是小众的 Bayesian Pruning 方法，考虑了对抗样本和正常样本的分布关系。提出了vulnerability来衡量某个权重的脆弱性来决定剪枝目标，为了保证原精度，正常样本的loss也会影响剪枝目标，二者做了个加权和。
