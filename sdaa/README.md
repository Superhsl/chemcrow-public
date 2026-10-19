# ChemCrow 龙芯 SDAA 本地推理

本目录在 LoongArch64 / SDAA 上运行官方 Molecular Transformer 反应预测命令行及官方 Flask /api/v1/run 路由。官方 ChemCrow 源码、模型算法和 CUDA 基线保持独立。完整的 OpenAI 驱动 Agent 不在本复现范围内。

前提：宿主机可使用 Docker、存在 /dev/tcaicard* 设备，并已加载 sdaa/locks/base-image.lock 指定且本地 image ID 一致的基础镜像。模型权重由构建入口按 sdaa/locks/source.lock 下载并校验；如网络受限，可预先将同哈希权重放入仓库同级的 models/molecular-transformer/。

在包含 sdaa/ 的源码仓库根目录执行：

~~~bash
./sdaa/build.sh
./sdaa/run.sh
~~~

构建入口检查 LoongArch64，下载或核对官方权重，校验基础镜像和三份 wheel，然后创建独立镜像、Python 3.12 venv 与容器 chemcrow-sdaa-repo。运行入口依次执行 SDAA 设备探针、官方分词器与 OpenNMT 命令行推理、官方 Flask 路由，并检查两个路径的产物排序。输入是 CCOCCCCO.CC(=O)Cl，beam size 10、n-best 5、max length 300、batch size 1；实际推理设备为 sdaa:0。

默认输出在仓库同级的 work/sdaa-repo/，包括 device-smoke.json、molecular-transformer/result.json、molecular-transformer/output.txt、official-api/result.json 及相关日志。运行入口遇到旧输出会退出，不会覆盖；构建入口遇到同名容器也会退出。另一次运行可在构建前同时设置新的绝对路径 CHEMCROW_WORK_DIR 和新的 CHEMCROW_CONTAINER_NAME。源码和权重只读挂载，输出目录可写，容器使用宿主机数字 UID/GID。

从干净检出开始的主机准备、完整命令、检查方式和限制见 [LOONG_SDAA_REPRO.md](LOONG_SDAA_REPRO.md)。具体适配点、依赖哈希及 CUDA/SDAA 输出和性能对比见 [SDAA_ADAPTATION.md](SDAA_ADAPTATION.md)。此前已验证的是独立 SDAA 工作副本；仓库内封装已在 LoongArch64/SDAA 上构建并跑通本地命令行与官方 Flask 路由。
