## 背景
SGLang进入训练框架之后，需要保持`enable_memory_saver=True`开启以便在rollout结束后释放显存给别的模块使用

## verl 完整的memory saver PR需要
- [x] memory saver找不到头文件安装失败 https://github.com/fzyzcjy/torch_memory_saver/pull/2
- [ ] 跨进程传tensor错误 https://github.com/sgl-project/sglang/pull/4565
- [ ] 更新verl-sglang镜像
- [ ] 在verl engine中默认开启memory saver
- [ ] 更新verl对sglang rollout的依赖

verl的相关pr：
- https://github.com/volcengine/verl/pull/732 尝试解决2，4
- https://github.com/volcengine/verl/pull/756 尝试解决4，5
