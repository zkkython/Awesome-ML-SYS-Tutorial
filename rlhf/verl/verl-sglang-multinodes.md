## 背景
目前的main上的并行写法等价于限制了`tp<=8`，为了支持deepseek相关的训练，需要verl engine的跨node支持

## Proposal
https://gist.github.com/fzyzcjy/01b851e045b970f16e63580b12dbf7ab

还在早期讨论阶段，预期来说，应该主要问题在于master addr和port怎么解决
- [ ] 获取ray driver进程的ip，显式传入
- [ ] 由verlengine自己获取
- [ ] 可能需要参考一下verl的vllm的处理
