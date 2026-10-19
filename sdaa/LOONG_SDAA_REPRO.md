# ChemCrow 龙芯 SDAA 完整复现

本教程从包含 sdaa/ 的干净检出运行官方 Molecular Transformer 本地反应预测。实际计算后端为 SDAA；输入由官方 SMILES tokenizer 在本地处理，无 MSA、模板或在线搜索。权重为官方 USPTO480k_model_step_400000.pt。最短命令见 [README.md](README.md)，适配问题与 CUDA 对比见 [SDAA_ADAPTATION.md](SDAA_ADAPTATION.md)。

## 1. 主机、源码与镜像

目标主机为 LoongArch64，有 Docker 使用权限和至少一个 /dev/tcaicard* 字符设备。基础镜像须已在本机加载：lasermpnn-sdaa321-loong:torch2.12.0-sdaa3.2.1-py312，本地 image ID 固定在 sdaa/locks/base-image.lock。该镜像无 RepoDigest，不能只凭标签确认镜像内容；构建脚本每次用 docker image inspect 比较完整 ID。镜像内 Python 3.12.13、PyTorch 2.12.0、torch_sdaa 3.2.1。可用 /opt/tecoai/bin/teco-smi 查看设备状态；该命令可能不在 PATH。

官方源码基线提交为 e7ebd5193334ac1d8dea137b635721c7cb470d33。从发布的仓库取得 URL 与完整适配提交 SHA，分别设置 CHEMCROW_REPO_URL 和 CHEMCROW_ADAPT_COMMIT；以下命令会固定检出该提交。现有包含 sdaa/ 的工作树可跳过检出步骤，从主机检查继续。

~~~bash
if [ -z "$CHEMCROW_REPO_URL" ] || [ -z "$CHEMCROW_ADAPT_COMMIT" ]; then
  printf '先设置 CHEMCROW_REPO_URL 和完整的 CHEMCROW_ADAPT_COMMIT。\n' >&2
  false
else
  git clone "$CHEMCROW_REPO_URL" chemcrow-public &&
    git -C chemcrow-public checkout --detach "$CHEMCROW_ADAPT_COMMIT" &&
    test "$(git -C chemcrow-public rev-parse HEAD)" = "$CHEMCROW_ADAPT_COMMIT"
fi
~~~

在检出的仓库根目录执行以下只读主机检查。若架构、设备或镜像 ID 不符，先准备正确环境；不要修改锁文件来绕过校验。

~~~bash
cd chemcrow-public
uname -m
id -u
id -g
docker info --format '{{.ServerVersion}}'
ls -l /dev/tcaicard*
grep '^commit=e7ebd5193334ac1d8dea137b635721c7cb470d33$' sdaa/locks/source.lock
base_image=$(sed -n 's/^image=//p' sdaa/locks/base-image.lock)
locked_id=$(sed -n 's/^image_id=//p' sdaa/locks/base-image.lock)
actual_id=$(docker image inspect "$base_image" --format '{{.Id}}')
test "$actual_id" = "$locked_id"
~~~

sdaa/scripts/build-and-start.sh 使用当前宿主机数字 UID/GID 构建并运行容器，映射现有全部 /dev/tcaicard* 字符设备。官方源码与权重只读挂载，仓库同级的 work/sdaa-repo/ 作为可写输出挂载到容器 /workspace/repro/work。模型运行时固定使用容器内 sdaa:0；设备数以实际主机为准。

## 2. 准备模型资产

权重来源及 Google Drive 文件 ID 见 sdaa/locks/source.lock，目标路径为仓库同级的 models/molecular-transformer/USPTO480k_model_step_400000.pt，SHA-256 为 1b610fee588a9632543605d28423ba589d63447d7921bbee36eac1b8a48de587。权重未纳入 Git；本包没有随权重提供单独许可文本，应遵守官方模型的使用条件。官方源码仓库的 LICENSE 为 MIT。

构建入口会在缺少权重时尝试下载，在权重已存在时先核对哈希。若 Google Drive 无法访问，可将从官方来源取得的同名文件预先放到上述位置；同名文件哈希不符时脚本退出并保留原文件。三份通用 wheel 已包含在 sdaa/vendor/，来源、许可证和 SHA-256 记录在 sdaa/locks/。

~~~bash
sha256sum -c sdaa/locks/wheels.sha256
./sdaa/scripts/download-model-weights.sh
repo_dir="$PWD"
(cd ../models/molecular-transformer && sha256sum -c "$repo_dir/sdaa/locks/model-weights.sha256")
~~~

官方示例输入在 sdaa/scripts/run-molecular-transformer.py 中固定为 CCOCCCCO.CC(=O)Cl，不需要数据库或外部 API。输出是五个排序后的产物 SMILES；该示例没有真值标签。

## 3. 构建环境与运行官方示例

在仓库根目录运行。若已按第 2 节下载，build.sh 会再次核对权重，不会重复下载。

~~~bash
./sdaa/build.sh
./sdaa/run.sh
~~~

sdaa/build.sh 调用 sdaa/scripts/download-model-weights.sh 和 build-and-start.sh；后者校验权重、wheel 与完整基础镜像 ID，并用 sdaa/docker/Dockerfile 构建独立镜像 chemcrow-sdaa:repo-torch2.12-sdaa3.2.1-py312、容器 chemcrow-sdaa-repo、Python venv /opt/venvs/chemcrow-sdaa-py312。镜像从基础镜像使用 torch、torch_sdaa 和 RDKit，离线安装 OpenNMT-py 2.2.0、TorchText 0.5.0 与 ConfigArgParse 1.8.0。环境与资产锁位于 sdaa/locks/。

sdaa/run.sh 调用 sdaa/scripts/run-local-examples.sh，依次检查矩阵乘法与设备随机数、运行官方 tokenizer 和 OpenNMT 推理、通过官方 Flask test_client 调用 /api/v1/run，并核对设备、HTTP 状态和产物排序。参数为 beam size 10、n-best 5、max length 300、batch size 1；未显式设置额外采样种子或 dtype。OpenNMT 的旧 CUDA 调用只在推理子进程中经 TORCH_SDAA_AUTOLOAD=cuda_migrate 映射到 sdaa:0。

若 chemcrow-sdaa-repo 容器已存在，build.sh 会退出而不覆盖；已停止的容器可用 docker start chemcrow-sdaa-repo 启动。run.sh 遇到已有输出会退出。另一次完整运行请在构建前同时选择新容器名和新输出目录，例如：

~~~bash
export CHEMCROW_CONTAINER_NAME=chemcrow-sdaa-repro-2
export CHEMCROW_WORK_DIR="$PWD/../work/sdaa-repro-2"
./sdaa/build.sh
./sdaa/run.sh
~~~

## 4. 检查结果与限制

默认输出目录为仓库同级的 work/sdaa-repo/。若设置了 CHEMCROW_WORK_DIR，按该绝对目录检查。run.sh 成功时打印 SDAA local examples: PASS，退出码为 0；若任一步失败，检查相应日志和非零退出码。

~~~bash
work_dir="$PWD/../work/sdaa-repo"
if [ -n "$CHEMCROW_WORK_DIR" ]; then work_dir="$CHEMCROW_WORK_DIR"; fi
ls -l "$work_dir/device-smoke.json" "$work_dir/molecular-transformer/output.txt" "$work_dir/molecular-transformer/result.json" "$work_dir/official-api/result.json"
python3 -B - "$work_dir" <<'PY'
import json
import pathlib
import sys
work = pathlib.Path(sys.argv[1])
device = json.loads((work / "device-smoke.json").read_text())
cli = json.loads((work / "molecular-transformer/result.json").read_text())
api = json.loads((work / "official-api/result.json").read_text())
assert device["sdaa_available"] and cli["device"] == "sdaa:0"
assert api["http_status"] == 200 and api["official_status"] == "SUCCESS"
assert api["same_ranked_products_as_cli"]
print("local inference and official API: PASS")
PY
~~~

日志位于 device-smoke.log、molecular-transformer/command.log、molecular-transformer/inference.log、official-api-command.log 和 official-api/bridge.log。默认样例的排名第一产物为 CCOCCCCOC(C)=O。独立 SDAA 工作副本已跑通本地命令行与官方 Flask 路由，并与 CUDA 基线得到逐字节相同的五个候选输出；本仓库新整理的 build.sh、run.sh 和挂载布局已按本教程在 LoongArch64/SDAA 上验证；本次输出保存在仓库同级 work/sdaa-repo/。

本流程不会运行完整的 OpenAI 驱动 ChemCrow Agent，也不执行搜索或生成蛋白结构。示例没有参考产物标签，因此输出一致不等于真实预测准确率。Docker 基础镜像必须与锁定的本地 ID 一致；换主机、镜像或驱动后需重新验证。性能测量方法与原始数据见 [SDAA_ADAPTATION.md](SDAA_ADAPTATION.md)。
