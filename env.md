# 超算集群用户手册摘录 —— 作业提交与集群环境（脱敏版）

> 摘录自某 ARM 架构超算集群内部用户手册，**已脱敏处理**：不包含具体单位名称、系统代号、内网地址等信息；示例中的节点名以 `cnXXXXX` 占位、队列名以 `{your_queue}` 占位。
> 手册中的命令、脚本模板（作业提交模板、大页/HBW 内存使用等）均完整保留。
> 未收录章节：前期准备（报备 MAC 地址、全盘扫描、接入集群、免密配置）、数据传输、Web 页面登录使用。

---

## 第一部分 集群环境

### 1. 系统与硬件环境

#### 计算节点配置

每个计算节点包含：

- 2 个 304 核、ArmV8 架构的 CPU（每个物理核为单线程），CPU 主频 1.55GHz
- 每个 CPU 具备 32GB HBM 内存，可访问 512GB DDR5 内存
- 节点间通过 200Gb RoCE 网络连接

#### 关键术语

| 术语 | 解释 |
|---|---|
| 作业调度器 | 集群调度系统，提供作业提交、队列管理、资源分配等功能 |
| 调度器 Web Portal | 调度器的图形界面，为用户访问集群资源提供方便易用的入口。用户使用用户名和密码登录，成功授权后获取系统访问权限 |
| 登录节点 | 用户接入集群的入口，用于代码编译、作业提交和轻量级文件操作。**严禁在此节点运行计算任务** |
| 计算节点 | 实际执行计算任务的服务器，通过调度系统统一分配资源 |
| 队列 | 作业被调度器调度时的通道，作业只有被提交到某个队列后才可以被调度器调度 |

### 2. 登录集群

使用任意 ssh 工具连接登录节点（地址以现场提供为准），输入集群账号密码即可接入集群。

连接到登录节点后：

```bash
dlogin    # 输入账号密码，完成登录认证授权
dqueue    # 查看当前账号可用队列
```

### 3. HPCKit 编译环境

HPCKit 包含编译器 clang、clang++、flang，并行库 hmpi，数学库 kml 等常用软件环境：

```bash
source /work_ssd/software/HPCKit/25.2.1/setvars.sh
```

### 4. 应用软件（module 加载）

#### 使用方式

集群已内置部分应用和工具/库，加载方式：

```bash
module use /work_ssd/software/modulefile/modules/
module load <app_name>/<app_version>-<hpckit_version>-<compiler_version>
```

#### 应用列表

| 分类 | 应用名 | 版本 |
|---|---|---|
| 地震学 | specfem3d | 4.1.1 |
| 第一性原理 | QE | 7.3.1 |
| 分子动力学 | DeePMD-kit | 3.0.1 |
| 分子动力学 | Lammps | 2023.8.2 |
| 分子动力学 | gromacs | 2024.2 |
| 流体力学 | OpenFOAM | V2412 |
| 气象海洋 | WRF | 4.6 |
| 生命科学 | Relion | 4.0.1 |
| 生命科学 | AlphaFold2 | 2 |
| 分子动力学 | SPONGE | 1.4 |
| 生命科学 | AlphaFold3 | 3.0.1 |
| 大语言模型 | deepseek | R1 |
| 第一性原理 | CP2K | 7.1 |
| 分子动力学 | NAMD | 3.0b6 |
| 流体力学 | PALABOS | 2.1r0 |
| 气象海洋 | CESM | 1.2.2 |
| 气象海洋 | NEMO | 4.2.2 |
| 气象海洋 | WAVEWATCH | 6.07.1 |
| 气象海洋 | CAMx | 7.31 |
| 生命科学 | BWA | 0.7.17 |
| 生命科学 | Augustus | 3.3.3 |
| 生命科学 | GATK | 4.0.0.0 |
| 生命科学 | Chaste | 2019.1 |
| 生命科学 | phonopy | 2.25.0 |
| 材料科学 | wannier tools | 2.7.1 |
| 生命科学 | phono3py | 3.2.0 |
| 气象海洋 | CESM | 2.2.2 |
| 气象海洋 | NCL | 6.3.0 |
| 深度学习 | xla | tensorflow2.13.0 |
| 深度学习 | tvm | 0.16.0 |
| 基础科研 | TensorFlow | 2.13.0 |
| 基础科研 | gulp | 6.2 |
| 气象海洋 | CDO | 1.9.8 |
| 基础科研 | Multiwfn | 3.8(dev) |
| 气象海洋 | NCO | 5.2.6 |
| 基础科研 | Wannier90 | 3.1.0 |
| 气象海洋 | ncview | 2.1.5 |
| 基础科研 | Anaconda3 | 2023.03 |
| 基础科研 | SCALAPACK | 2.0.2 |
| 基础科研 | OpenBLAS | 0.3.18 |
| 基础科研 | boost | 1.72 |
| 基础科研 | CMAKE | 3.26.0 |
| 基础科研 | BLAS | 3.8.0 |
| 气象海洋 | gsl | 2.6 |
| 基础科研 | FFTW | 3.3.8 |
| 基础科研 | Lapack | 3.8.0 |
| 深度学习 | PyTorch | 2.9.0 |
| 基础科研 | KML | 2.3.0 |
| 基础科研 | bishengcompiler | 4.0.0 |
| 基础科研 | bishengJDK | 11.0.23+11 |
| 基础科研 | HMPI | 2.3.0 |

详细列表可在 `module use /work_ssd/software/modulefile/module` 之后通过 grep 查找。以 vasp 为例（`module list` 很长，建议用 grep 查找）：

```bash
module use /work_ssd/software/modulefile/modules
module av | grep vasp
module load vasp/6.3.2-hpckit25.1.0.SPC001-bisheng4.2.0.2.B002-hmpi25.1.0.SPC001
```

### 5. Python 离线包安装流程（以 numpy 为例）

#### ① 准备 miniconda

从官方渠道下载 aarch64 版安装包（建议：使用 latest 会默认 python3.13，如不需要该版本可下载对应版本的安装包），上传到集群后执行：

```bash
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
```

如果不需要使用复杂环境，可以默认加载 miniconda（安装过程中的相关选择项选 yes 即可）。

#### ② 下载 wheel 文件（本地执行）

```bash
pip download numpy -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --only-binary=:all: \
    --platform manylinux_2_27_aarch64 --platform manylinux2014_aarch64 \
    --python-version 313
```

- `numpy` 可替换成需要的其他包
- `--python-version` 后的数字需与安装 miniconda 时选择的 python 版本一致
- 其余参数请不要调整

下载完成后（如 `numpy-2.4.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`），将 whl 文件上传至集群。

#### ③ 集群安装

在上传 whl 的目录下执行：

```bash
source miniconda3/etc/profile.d/conda.sh
conda activate base
pip install --no-index --find-links=./ *.whl
```

安装成功会有相应的成功输出。

### 6. 公共工具

#### huge page（透明大页，含 HBM/DDR 大页）

**方式一：交互式手动设置**

```bash
# 提交交互式作业（示例节点以 cnXXXXX 占位）
dsub -q {your_queue} -nl "cnXXXXX" -nn 2 -rpn 1 -I bash

# 进入工具目录
cd /work_ssd/software/performance_tune

# 运行大页脚本（参数依次为 DDR/HBM 各规格大页数量，见下方参数说明）
./root_proxy hugepage.sh 0 8192 0 0 0 0 0 0
```

**方式二：作业脚本中设置**

```bash
#!/bin/bash
#DSUB -n hugepage
#DSUB --mpi hmpi
#DSUB -q {your_queue}
#DSUB -nl "cn[XXXXX-XXXXY]"
#DSUB -nn 2
#DSUB -rpn 32
#DSUB -x job
#DSUB -o n1_out_%J.log
#DSUB -e n1_err_%J.log
### 以上为作业头，分配节点2个，每个节点32个资源副本，合计64个资源分布，且每节点独占

WORK_DIR=$(pwd)
{
set -x
### 进行节点的大页设置
### 步骤1：进入脚本目录
cd /work_ssd/software/performance_tune
### 步骤2：设置大页(8192 * 2M)，把需要配置的节点都加入 drun 的 -nl 列表
export MASK_NODE=$(( 2#1111111111111110))
drun -nl "cn[XXXXX-XXXXY]" -N 2 -rpn 1 ./root_proxy hugepage.sh 0 8192 0 0 0 0 0 0 ${MASK_NODE}
set +x
}
cd ${WORK_DIR}

### MPI 程序主体，可以透明使用透明大页
{
source env.sh
mpirun -np $((32 * 2)) \
    --mca coll ^ucg -mca pml ucx -mca btl ^vader,tcp,openib,uct -mca io romio321 \
    --bind-to core --map-by socket --rank-by core -map-by ppr:32:node:pe=1 \
    -x UCX_TLS=sm,rc \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x UCX_NET_DEVICES=roceroh0:1 HelloWorld
}
```

**关闭（回收大页内存）：**

```bash
{
### 步骤1：进入脚本目录
cd /work_ssd/software/performance_tune
### 步骤2：回收大页内存
export MASK_NODE=$(( 2#1111111111111111))
drun -nl "cn[XXXXX-XXXXY]" -N 2 -rpn 1 ./root_proxy hugepage.sh 0 0 0 0 0 0 0 0 ${MASK_NODE}
}
```

**参数说明（前 4 个为 DDR 各规格、后 4 个为 HBM/HBW 各规格）：**

```text
Usage:   ./root_proxy hugepage.sh <ddr_64k> <ddr_2m> <ddr_32m> <ddr_1g> <hbm_64k> <hbm_2m> <hbm_32m> <hbm_1g> NODE_MASK [quiet]
Example: ./root_proxy hugepage.sh 16 8 2 4 8 4 1 2 NODE_MASK quiet
# quiet: 任意非空值将抑制所有输出
```

> 提示：`NODE_MASK` 为 16 位二进制掩码（对应节点 16 个 NUMA），按需置 0/1 选择生效的 NUMA。

#### drop cache

**方式一：交互式手动设置**

```bash
# 提交交互式作业（示例节点以 cnXXXXX 占位）
dsub -q {your_queue} -nl "cnXXXXX" -nn 2 -rpn 1 -I bash

# 进入工具目录
cd /work_ssd/software/performance_tune/

# 运行脚本
./root_proxy drop_cache.sh
```

**方式二：作业脚本中设置**

```bash
### 在作业脚本中添加如下代码
cd /work_ssd/software/performance_tune
drun -nl "cn[XXXXX-XXXXY]" -N 2 -rpn 1 ./root_proxy hugepage.sh 0 0 0 0 0 0 0 0 ${MASK_NODE}
```

---

## 第二部分 作业提交

### 1. 交互式提交方式

交互式作业指作业提交后不立即退出，CLI 会持续等待并显示部分作业的状态变化及实时输出。当作业需要用户输入时，用户可在当前 CLI session 输入，作业继续运行直到结束。**该作业在 CLI session 断开后会被强制结束。**

命令格式：

```bash
dsub -I -q {your_queue} bash
```

`{your_queue}` 为当前账号可用队列（可用 `dqueue` 查询）。

### 2. 作业脚本提交

编写作业脚本文件，修改节点数、进程数、线程数等需求（**不需要改变 rankfile 生成和 mpirun 运行参数**），添加可执行权限，使用命令 `dsub -s 脚本` 提交作业。

**查看输出与连接执行节点：**

- 使用 `dattach` 命令可以连接到作业执行节点（仅支持连接 RUNNING 状态作业），使用 `exit` 命令退出。
- 单节点作业：`dattach jobid` 连接到作业执行节点。
- 多节点作业：`dattach jobid` 默认连接到第一个节点，`dattach -en "node" jobid` 连接到指定节点。

**普通作业脚本内容参考：**

```bash
#!/bin/bash
#DSUB -n test
#DSUB -N 4
#DSUB -q {your_queue}
#DSUB -o output
date
echo "this is script job"
```

**MPI 作业脚本内容参考：**

```bash
#!/bin/bash
#DSUB --mpi hmpi
#DSUB -q {your_queue}
#DSUB -n {your_job_name}
##DSUB -N 608
#DSUB -rpn 16
#DSUB -nn 1
#DSUB -oo out.%A
#DSUB -eo err.%A

export OMP_NUM_THREADS=36
np=16
EXE={your_exe}
source /work_ssd/software/HPCKit/25.2.1/setvars.sh > /dev/null

RANK=0
for NODE in `awk '{print $1}' ${CCS_ALLOC_FILE} | sort`
do
  for NUMA in {0..15}
  do
    echo rank $RANK=$NODE slot=$(($NUMA * 38))-$(($NUMA * 38 + 35))
    RANK=$(($RANK + 1))
  done
done > $PWD/${CCS_JOB_ID}-rankfile

echo launch on $HOSTNAME
time \
mpirun \
  -x PATH -x LD_LIBRARY_PATH -x UCX_TLS=sm,rc \
  --mca rmaps_rank_file_physical true \
  --mca grpcomm_direct_priority 100 \
  --mca coll_tuned_use_dynamic_rules true \
  -x UCX_RNDV_THRESH=512K \
  -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
  -x BINDPROCNIC_NOSHOW=1 \
  -x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
  --mca plm_rsh_agent /usr/bin/ssh \
  -np $((CCS_TASK_REPLICA)) \
  --rankfile $PWD/${CCS_JOB_ID}-rankfile \
  $EXE
```

**dsub 参数说明：**

| 参数 | 说明 |
|---|---|
| `-n` \| `--name` | 提交作业时，指定作业的名称 |
| `-nl` \| `--node-list` | 提交作业时，指定节点或资源池偏好 |
| `-A` \| `--account` | 提交作业时，提交作业到指定 Account 下 |
| `-q` \| `--queue` | 提交作业时，提交作业到指定队列下 |
| `-N` \| `--replica` | 提交作业时，指定作业的资源副本数量 |
| `-o` \| `--output` | 指定输出文件名称 |
| `-e` \| `--error` | 提交作业时，重定向该作业的错误输出日志路径 |
| `-R` \| `--resource` | 提交作业时，指定作业的资源需求 |
| `-mR` \| `--min-resource` | 提交作业时，指定作业的最小资源需求 |
| `--mpi` | 提交 MPI 作业时，指定作业的类型，调度器会根据指定的作业类型来调度执行 MPI 作业 |
| `-rpn` \| `--replica-per-node` | 提交作业时，指定作业在每个节点运行的最大任务数 |
| `-nn` \| `--nnodes` | 提交作业时，指定作业运行的节点数 |

### 3. 作业任务运行方式（核隔离机制）

#### 机制介绍

为减少操作系统后台任务对计算任务的影响，本集群使用 `isolcpu` 启动参数，启用了核隔离（CPU isolation）机制，集群中 CPU 核心分两类：

**计算核心：**

- 用途：专用于处理计算任务，不处理周期性时钟中断、不处理 RCU 回调等，最大限度减少系统噪声
- 核心列表：`0-36,38-74,76-112,114-150,152-188,190-226,228-264,266-302,304-340,342-378,380-416,418-454,456-492,494-530,532-568,570-606`

**系统核心：**

- 用途：作为 housekeeping 系统核，处理系统服务、各系统 agent 进程、公共服务、处理中断等
- 核心列表：`37,75,113,151,189,227,265,303,341,379,417,455,493,531,569,607`（每个节点的 16 个 NUMA 中，每 NUMA 的最后一个核心）

在此配置下，**命令行执行的进程会运行在系统核上**，只能感知到系统核范围内的 CPU 资源。如果想要将应用运行在计算核上，需要感知隔离核，必须显式地指定需要使用的计算核。

#### 任务拉起方式

**taskset 方式**

原本使用 taskset 执行的作业，在核隔离场景下，需要注意 `-c` 指定的 CPU 范围是否包含系统核。**禁止使用 `taskset -c 0-607` 来启动任务。** 如需使用全部计算核，需显式指定：

```bash
taskset -c 0-36,38-74,76-112,114-150,152-188,190-226,228-264,266-302,304-340,342-378,380-416,418-454,456-492,494-530,532-568,570-606 ./your_app
```

**numactl 方式**

与 taskset 不同，numactl 自己绑核时范围也不能有隔离核。可以使用以下几种方式解决（**建议使用前两种**）：

1. 使用 `taskset -c xx-xx` 来替换
2. 在 numactl 前添加 `taskset -c xx-xx`
3. numactl 在 `-C` 参数前增加 `--all` 参数

示例：

```bash
taskset -c 0-31 numactl -C 0-31 ./your_app
taskset --all -C 0-31 ./your_app   # 原文示例；按上文说明即 numactl 增加 --all 参数的写法
```

**rankfile 方式**

rankfile 示例（每 rank 占一个 37 核的计算核心段）：

```text
rank 0=cnXXXXX slot=0-36
rank 1=cnXXXXX slot=38-74
rank 2=cnXXXXX slot=76-112
rank 3=cnXXXXX slot=114-150
rank 4=cnXXXXX slot=152-188
rank 5=cnXXXXX slot=190-226
rank 6=cnXXXXX slot=228-264
rank 7=cnXXXXX slot=266-302
rank 8=cnXXXXX slot=304-340
rank 9=cnXXXXX slot=342-378
rank 10=cnXXXXX slot=380-416
rank 11=cnXXXXX slot=418-454
rank 12=cnXXXXX slot=456-492
rank 13=cnXXXXX slot=494-530
rank 14=cnXXXXX slot=532-568
rank 15=cnXXXXX slot=570-606
```

命令示例：

```bash
mpirun -np 4096 --rankfile rk16 --mca rmaps_rank_file_physical true \
    -x PATH -x LD_LIBRARY_PATH ./your_app
```

#### 作业任务运行方式（线程分布）确认参考

会拉起新线程/进程的常见方式：

- **python**：使用 `threading.Thread`（底层通过 pthread_create 实现）、`multiprocessing.Process`（fork 模式）拉起子进程、使用 `subprocess.Popen` 或 `os.system` 执行新的可执行文件
- **C 语言**：`fork()`、`pthread_create()`、`vfork()`、`clone()`

系统监测手段层面排查：

```bash
# 运行核号确认：确认各进程和线程所运行的核（第9列数据）
ps -eLF | grep $your_app

# 系统调用确认：抓取所有系统调用，在其中寻找关键字（如 clone 等）
strace ./your_app        # 拉起应用时
strace -p $app_pid       # 应用运行时
```

参考输出：

- 起进程时会抓到类似：`clone(SIGCHLD, 0)`
- 起线程时会抓到类似：`clone(CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | ...)`

#### FAQ

**1. 作业运行后，均跑在系统核（37、75、113、151、189、...、607）上**

- 问题根因：在核隔离环境中，命令行运行的默认核为系统核，在未正确指定使用计算核时，会出现该问题
- 解决方案：参考上述「任务拉起方式」中的描述，启动业务进程

**2. 核隔离场景下 pthread 绑核提交示例**

方式一：通过 `LD_PRELOAD` 环境变量，对当前进程及加载的动态库生效：

```bash
LD_PRELOAD=/work_ssd/system/prehook/libpthread_hook.so \
taskset -c 0-36,38-74,76-112,114-150,152-188,190-226,228-264,266-302,304-340,342-378,380-416,418-454,456-492,494-530,532-568,570-606 ./your_app
```

方式二：修改源码，通过 `pthread_setaffinity_np` 进行绑核。demo 代码：

```c
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#define NUM_THREADS 32

// 线程函数：打印线程ID和运行的CPU核心
void* hello_thread(void* arg)
{
    int cpu = *(int*)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    printf("thread %d cpu=%d\n", cpu, sched_getcpu());
    sleep(5);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    printf("=== pthread Hello World (32 threads + CPU affinity) ===\n");
    printf("Total CPU cores: %d\n\n", (int)sysconf(_SC_NPROCESSORS_ONLN));
    // 创建32个线程
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i + 1;
        int rc = pthread_create(&threads[i], NULL, hello_thread, &thread_ids[i]);
        if (rc != 0) {
            fprintf(stderr, "Error: failed to create thread %d\n", i);
            return 1;
        }
    }
    printf("All 32 threads created. Waiting for completion...\n\n");
    // 等待所有线程结束
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("\nAll 32 threads finished. Main thread exiting.\n");
    return 0;
}
```

#### HMPI 使用参考

**MPI UCX_TLS 参数**

当进程规模大于等于 4096（节点 ppn 为 16）的场景下，使用 alltoall 等多对多集合通信时，手动指定 UCX_TLS 为 `rc` 会超出 qp 硬件上限，需要手动指定为 `sm`、`ud`，或者不指定 TLS 参数（hmpi 会自动选择 TLS 协议）。

**集合通信算法参数**

当前的默认算法可以支持大规模应用场景。如果想修改集合通信算法进行深度调优，可以通过参数选择使用 UCG 算法还是 tuned 算法。

使能 tuned 算法示例：

```bash
mpirun -np 4096 --rankfile rk16 --mca rmaps_rank_file_physical true \
    -x PATH -x LD_LIBRARY_PATH \
    --mca coll_tuned_use_dynamic_rules true \
    --mca coll_tuned_allgather_algorithm 4 ./exe   # 选择 allgather 的算法4
```

使能 ucg 算法示例：

```bash
mpirun -np 4096 --rankfile rk16 --mca rmaps_rank_file_physical true \
    -x PATH -x LD_LIBRARY_PATH \
    --mca coll_ucg_enable_coll gatherv \
    -x UCG_PLANC_UCX_GATHERV_ATTR=I:4 ./exe        # 选择 gatherv 的算法4
```

### 4. 集群基础命令（状态查询 & 作业管理）

常用作业查询、提交、作业管理命令：

```bash
# 查询队列内节点信息
dinfo -q your_queue

# 查询全部队列信息
dinfo -q all

# 查询指定节点的列表信息
dnode cnXXXXX
dnode "cnXXXXX cnYYYYY"
dnode "cn[XXXXX-YYYYY,ZZZZZ]"

# 作业提交 - 指定 prehook
dsub --prehook "~/hook.sh" your_exe

# 作业提交 - 指定队列
dsub -q queuename your_exe

# 作业提交 - 指定目标节点
dsub -q queuename -nl 'cn[XXXXX-YYYYY]' your_exe

# 作业提交 - 指定标准输出和错误输出
dsub -q queuename -o ~/job.log -e ~/job.error your_exe

# 作业查看
djob jobid
djob -l jobid

# 作业终止
dkill jobid

# 查看应用日志目录
djob -l jobid | grep RedirectOutputPath

# 访问正在运行作业的节点
dattach -l jobid | -en exec_nodeid
```

### 5. 调度脚本常用参数示例（#DSUB）

脚本文件中 `#DSUB` 为调度器选项的标识符；如不需要可以使用两个 `#` 的方式注释（`##DSUB`），或直接删除该行。

| 参数 | 说明 |
|---|---|
| `#DSUB --mpi hmpi` | 指定 MPI 类型为 Hyper MPI |
| `#DSUB -x job` | 提交独占作业，独占作业运行的节点上不能同时执行其他作业 |
| `#DSUB -q {your_queue}` | 指定作业提交到的队列 |
| `#DSUB -n {your_job_name}` | 指定任务名称 |
| `#DSUB -N 608` | 指定作业运行需要的资源份数（一份资源是由 `-R` 指定的多种资源的组合），默认一份资源会启动一个任务。缺省规则：<br>• 未指定该参数且未指定 nn 和 rpn：缺省值为 N=1<br>• 未指定该参数但同时指定了 nn 和 rpn：默认 N = nn(min) × rpn<br>• 未指定该参数但仅指定了 rpn：默认 N=rpn 且 nn=1<br>• 未指定该参数但仅指定了 nn：默认 N=nn 且 rpn=1 |
| `#DSUB -rpn 16` | 指定作业在每个节点分配最大资源份数。如果作业的 -N 资源副本数小于该值，则该作业会分配到一个节点上运行 |
| `#DSUB -nn 1` | 指定作业分布的节点数。支持指定最小、最大运行的节点数，标准用法为 `-nn "min,max"`；如果 min=max，可简化为 `-nn x` |
| `#DSUB -R "cpu=4;mem=1G"` | 指定每份资源的具体资源需求；cpu 默认值为 1，mem 默认值为 128、默认单位 MB |
| `#DSUB -oo out.%A` | 指定覆盖式重定向标准输出路径 |
| `#DSUB -eo err.%A` | 指定覆盖式重定向错误输出路径 |
