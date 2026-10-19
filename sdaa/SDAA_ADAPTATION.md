# ChemCrow 龙芯 SDAA 适配与验证报告

## 1. 结论摘要

- 官方本地示例：独立 SDAA 工作副本中的 Molecular Transformer 命令行与官方 Flask /api/v1/run 均已运行并通过，共 2/2；实际后端为 SDAA，结果 JSON 记录 sdaa:0。
- 特征准备：本任务无 MSA 或模板；使用官方 tokenizer 在本地处理输入，未执行在线搜索。
- 源码改动：官方 ChemCrow 算法和 sdaa/ 以外源码均无修改。新增的 Docker、依赖、包装器和运行脚本仅在本仓库的 sdaa/；CUDA 基线和 SDAA 官方源码均为 e7ebd5193334ac1d8dea137b635721c7cb470d33。适配提交 SHA 以发布的仓库历史为准。
- 精度对照：一条官方反应样例在 CUDA 与 SDAA 上各运行 6 次，12/12 个输出文件逐字节一致，五个产物排序一致；输出 SHA-256 为 c4eadd1af6aa49e3d6ee0244749c9b7d45dc4e34b057398697d936731cd5e588。该样例没有真值标签，不提供整体预测准确率结论。
- 性能对照：2026-10-14 各预热 1 次、正式计时 5 次；CUDA H100 GPU 3 中位数 7.906 秒，SDAA 设备 0 中位数 21.632 秒，同一端到端计时口径下 SDAA 耗时为 CUDA 的 2.736 倍。
- 范围与遗留问题：完整 OpenAI 驱动的 ChemCrow Agent 未运行；只验证上述本地推理路径。上述多轮对照结果来自原独立 SDAA 工作副本；本仓库封装也已单独构建并通过本地命令行与官方 Flask 路由检查。原始输出和逐轮计时保留在两个基线项目的 work/ 目录，不随源码提交。

## 2. 环境与资产

| 环境 | 架构/设备 | 镜像 digest 或本地 ID | Python/框架/插件 | 源码 commit |
| --- | --- | --- | --- | --- |
| CUDA 基线 | x86_64；NVIDIA H100 80GB，物理 GPU 3，容器内 cuda:0 | pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 | Python 3.11.13；PyTorch 2.8.0+cu128；CUDA 12.8；Conda molecular-transformer | e7ebd5193334ac1d8dea137b635721c7cb470d33 |
| 龙芯适配 | LoongArch64；TECO_AICARD_01，sdaa:0 | lasermpnn-sdaa321-loong:torch2.12.0-sdaa3.2.1-py312；本地 ID sha256:3e425da47a74a03b589f8e1c18a67dcb90c8bcbde977cc1df09fa01209b7a524 | Python 3.12.13；PyTorch 2.12.0；torch_sdaa 3.2.1；独立 venv | e7ebd5193334ac1d8dea137b635721c7cb470d33 |

两端官方源码提交相同。CUDA 环境锁位于 CUDA 基线项目的 locks/molecular-transformer-conda-linux-64.lock 和 locks/molecular-transformer-installed-pip-freeze.txt；SDAA 环境锁位于 sdaa/locks/base-image.lock 和 sdaa/locks/venv-pip-freeze.txt。官方模型 USPTO480k_model_step_400000.pt 的来源见 sdaa/locks/source.lock，SHA-256 为 1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587；两端权重路径分别在各自项目的 models/molecular-transformer/。输入来源、路径和哈希见第 5 节。本任务无 MSA、模板或在线搜索资产；SDAA 基础镜像无 RepoDigest，其影响见第 3.3 节。

## 3. 必要修改与依赖

### 3.1 源码适配

| 文件/行或 patch | 原问题 | 最小改动 | 对算法或数值的影响 |
| --- | --- | --- | --- |
| sdaa/scripts/run-molecular-transformer.py | 旧版 OpenNMT 使用 torch.cuda 与 --gpu 0；SDAA 设备不会被原命令直接选中。 | 校验 torch.sdaa 可用性；只在 OpenNMT 子进程设置 TORCH_SDAA_AUTOLOAD=cuda_migrate，并沿用 --gpu 0。 | 官方 tokenizer、模型、beam search 与超参数不变；已测结果逐字节一致。 |
| sdaa/bin/onmt_translate、sdaa/scripts/run-official-api.py | 官方 Flask app.py 调用 onmt_translate 未传 --gpu 0。 | PATH 中的同名包装器补 GPU 参数及 SDAA 兼容环境；通过官方 Flask test_client 调用原路由。 | app.py 未修改；接口返回 HTTP 200、SUCCESS。 |
| sdaa/scripts/run-molecular-transformer.py | 新版 PyTorch 默认 checkpoint 加载策略与旧模型不兼容。 | 模型 SHA-256 校验后，仅在推理子进程设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1。 | 模型权重和计算逻辑不变。 |
| sdaa/scripts/device-smoke.py | 需核实设备、矩阵乘法和设备随机数。 | 检查 SDAA 设备、矩阵乘法、相同种子的重复性。 | 不参与正式模型预测；已测路径未遇到不支持算子。 |

官方仓库中的 ChemCrow 源码无改动；本适配没有算法 patch。改动前提交为 e7ebd5193334ac1d8dea137b635721c7cb470d33。

### 3.2 第三方依赖（确有必要时）

| 包/程序及版本 | 必要性和兼容问题 | 来源/许可证 | 构建命令或 wheel 标签 | SHA-256 |
| --- | --- | --- | --- | --- |
| OpenNMT-py 2.2.0 | 官方 Molecular Transformer 的推理框架；目标镜像未提供此旧版本。 | 来源见 sdaa/locks/opennmt-wheel.source；MIT。 | py3-none-any；离线、无依赖安装。 | ea8537b2c9bece7ce0fa2988b556fc18ae1ff657aa941e6be03af86ceef08ece |
| ConfigArgParse 1.8.0 | OpenNMT 命令行参数解析所需。 | 来源见 sdaa/locks/configargparse-wheel.source；MIT。 | py3-none-any；离线、无依赖安装。 | bb25b307c3cd46a3e868e7f7aa51487eb8003a9f01cac3ae129b66ec26eb794c |
| TorchText 0.5.0 | 与已固定的旧版 OpenNMT 推理路径配套。 | 来源见 sdaa/locks/torchtext-wheel.source；BSD 3-Clause。 | py3-none-any；离线、无依赖安装。 | 1caed2e155c45b4885daedb735b0f41e2f86ecd9dc788f75f824683bf1645f67 |

三个 wheel 已打包在 sdaa/vendor/，合计 324,562 字节，均为通用 Python wheel；不需要编译 LoongArch ABI 扩展。Dockerfile 先运行 sha256sum -c sdaa/locks/wheels.sha256，再执行 pip install --no-index --no-deps。

### 3.3 容器与环境

SDAA 基础镜像无 RepoDigest，sdaa/scripts/build-and-start.sh 在每次构建前读取 docker image inspect 的完整本地 ID 并与锁文件比较。Dockerfile 在独立 Python 3.12 venv 中安装三份 wheel；镜像由当前宿主机数字 UID/GID 构建和运行。构建脚本显式映射现有 /dev/tcaicard* 字符设备，将源码和模型以只读方式挂载、输出目录以可写方式挂载。原独立 SDAA 工作副本已完成设备推理验证。容器与依赖定义在 sdaa/docker/Dockerfile，来源和校验文件在 sdaa/locks/。构建入口为 sdaa/scripts/build-and-start.sh；若同名容器已存在，脚本会退出而不覆盖，已停止的容器可使用 docker start chemcrow-sdaa-repo 启动。

## 4. 遇到的问题

| 问题/触发命令 | 定位原因 | 处理方法与修改位置 | 重跑结果或遗留风险 |
| --- | --- | --- | --- |
| 旧 OpenNMT 的 torch.cuda、--gpu 0 设备调用 | 框架接口以 CUDA 命名。 | 在 sdaa/scripts/run-molecular-transformer.py 的子进程中启用 cuda_migrate。 | 通过：原 SDAA 工作副本 work/sdaa/molecular-transformer/result.json 记录 sdaa:0，输出与 CUDA 一致；其他算子路径未验证。 |
| 官方 Flask 路由未显式指定 GPU | app.py 的 onmt_translate 命令缺少设备选项。 | sdaa/bin/onmt_translate 包装原命令；sdaa/scripts/run-official-api.py 调用原路由。 | 通过：原 SDAA 工作副本 work/sdaa/official-api/result.json 记录 HTTP 200、SUCCESS、排序一致；新封装仍需复测 PATH 优先级。 |
| 旧 checkpoint 与 PyTorch 新加载策略 | 权重加载默认行为变更。 | 先校验权重哈希，再于推理子进程设置 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1。 | 通过：原 SDAA 工作副本 work/sdaa/molecular-transformer/inference.log 及结果文件显示推理完成；仅对固定哈希的官方权重使用该开关。 |
| 官方 setup.py 的 Python 约束与基础镜像 Python 3.12 不一致 | 当前镜像提供 Python 3.12。 | sdaa/docker/Dockerfile 建独立 venv，仅安装本地推理所需依赖。 | 本地 CLI 与 Flask 路由在原工作副本通过；完整 Agent 未验证。 |

官方 beam search 样例的 6 次输出一致。没有记录到已测路径的不支持算子。

## 5. 精度分析

两端均输入 CCOCCCCO.CC(=O)Cl，官方 tokenizer 生成的输入文件 SHA-256 均为 bd20855e98da7cb6930f70582056481f809efa4db4803736a290c952a060c233。权重 SHA-256 均为 1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587；beam size 10、n-best 5、max length 300、batch size 1。CUDA 与 SDAA 分别在各自项目中调用官方 tokenizer 和 OpenNMT；没有额外采样种子参数，设备探针单独检查了随机数重复性。

对比方法为逐个比较同日两端 run-0 至 run-5 的 output.txt 哈希，并核对排序后的五个产物。使用的逐字节一致判据是查看结果后的核对标准，不是预先注册的精度阈值。原始输出分别位于 CUDA 基线项目的 work/benchmark-cuda-gpu3-20261014-103629/ 和原 SDAA 工作副本的 work/sdaa/benchmark-sdaa-20261014-113001/；对应 run-N.result.json 记录设备与参数。

| 输出字段/任务指标 | 比较方法 | CUDA | 龙芯 | 差异与判据 |
| --- | --- | ---: | ---: | --- |
| 输入文件 | SHA-256 | bd20855e98da7cb6930f70582056481f809efa4db4803736a290c952a060c233 | 相同 | 0；一致 |
| 模型权重 | SHA-256 | 1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587 | 相同 | 0；一致 |
| 6 轮输出文件 | 每轮 SHA-256 | 6/6 为 c4eadd1af6aa49e3d6ee0244749c9b7d45dc4e34b057398697d936731cd5e588 | 6/6 为同一哈希 | 12/12 逐字节一致 |
| 五个产物及排序 | 按名次比较规范化 SMILES | 5 个 | 同样 5 个 | 5/5 一致 |
| 官方 Flask 路由 | HTTP 状态、业务状态、排序产物 | 未纳入本次跨平台接口对比 | 200、SUCCESS、与 SDAA CLI 一致 | 仅证明 SDAA 接口通过 |

五个排序产物依次为 CCOCCCCOC(C)=O、CCOCCCCOCC、[Cl-]、CCOCCCCSC(C)=O、CCOCCCC(C)=O。该单条反应没有真值标签，不能计算预测准确率。

## 6. 性能分析

计时窗口为宿主机调用 docker exec 到 Python wrapper 退出的 Bash 实时时间，包括 Python 启动、模型哈希校验、模型加载、tokenizer、OpenNMT 推理及输出写盘；不是纯设备计算或吞吐率。beam size 10、n-best 5、batch size 1；CPU 线程数未单独锁定。每端预热 1 次，正式计时 5 次。没有记录独立的显式设备同步点，进程退出是本次端到端计时终点。输入与权重同第 5 节。

| 平台/后端 | 次数 | 中位数（s） | 范围（s） | 原始数据 |
| --- | ---: | ---: | --- | --- |
| CUDA H100 物理 GPU 3 | 5 | 7.906 | 7.729–9.169 | CUDA 基线项目 work/benchmark-cuda-gpu3-20261014-103629/run-1.seconds 至 run-5.seconds |
| 龙芯 SDAA 设备 0 | 5 | 21.632 | 21.363–22.546 | 原 SDAA 工作副本 work/sdaa/benchmark-sdaa-20261014-113001/run-1.seconds 至 run-5.seconds |

正式五轮平均耗时分别为 CUDA 8.201 秒、SDAA 21.866 秒。CUDA 预热为 8.389 秒，正式五轮依次为 9.169、7.906、7.729、7.882、8.318 秒。SDAA 预热为 22.184 秒，正式五轮依次为 22.164、22.546、21.363、21.623、21.632 秒。CUDA 中位数 / SDAA 中位数 = 0.3655；等价地，SDAA/CUDA 耗时比 = 2.736，当前样例的 CUDA 端到端执行更快。

## 7. 复现命令

已验证的 CUDA 基线关键命令，在 CUDA 基线项目根目录执行；更换物理 GPU 时修改 CUDA_VISIBLE_DEVICES：

    docker compose up -d --no-deps molecular-transformer
    sha256sum -c locks/model-weights.sha256
    docker exec -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES=3 chemcrow-molecular-transformer python -B /workspace/repro/scripts/run-molecular-transformer.py

已验证的原 SDAA 工作副本关键命令，在该工作副本根目录执行：

    sha256sum -c locks/model-weights.sha256
    docker exec -e PYTHONDONTWRITEBYTECODE=1 chemcrow-sdaa python -B /workspace/repro/sdaa/scripts/run-molecular-transformer.py

同日原始计时文件路径见第 6 节，每轮输出和日志位于同名 run-N.output.txt、run-N.result.json、run-N.inference.log。SDAA 官方路由结果位于原 SDAA 工作副本的 work/sdaa/official-api/result.json。上述命令只重跑单次推理；完整批量计时需按各基线项目已保存的基准脚本或逐轮执行并保存原始记录。

本仓库 sdaa/ 封装已验证的复现命令如下；本次输出保存在仓库同级的 work/sdaa-repo/：

    ./sdaa/build.sh
    ./sdaa/run.sh

默认权重目录为本仓库同级的 models/molecular-transformer/，新输出目录为本仓库同级的 work/sdaa-repo/；也可设置绝对路径的 CHEMCROW_MODEL_DIR、CHEMCROW_WORK_DIR。新输出应包含 molecular-transformer/result.json、official-api/result.json 和 device-smoke.json。权重、轮次日志、备份、虚拟环境与私有配置不纳入 Git。完整从干净检出到结果检查的步骤见 [LOONG_SDAA_REPRO.md](LOONG_SDAA_REPRO.md)。build.sh 遇到同名容器会退出；run.sh 遇到旧输出会退出。
