# kp920 单核缓存带宽

测试程序：cache_bandwidth.cpp；运行：

```sh
bash test/run_cache_bandwidth.sh          # 16 KiB 到 64 MiB，7轮，每样本约100ms
bash test/run_cache_bandwidth.sh confirm  # 代表工作集确认，9轮，每样本约250ms
```

运行覆盖项目 result.out/error.err，临时可执行文件结束后删除。
CPU 固定0，先初始化并预热，再多次完整遍历。不同工作集和读/拷贝测试的顺序随机化。
原始样本全部保留在同一 result.out，SUMMARY 是按 GB/s 排序后的中位样本，
该行的 PMU 指标来自同一样本，不是各自独立挑选的最佳数字。

## 与 cpufb 的关系

纯读采用 cpufb/arm64/asm/_LDP_.S 的 load_ldp_kernel：每次内循环16条LDP，
实际读512字节。测试文件 cache_load_cpufb.S 保留这个负载，另加 d8-d15 的
ABI保存恢复，避免破坏C++调用者。这是GPL测试代码（许可证见cpufb_LICENSE），
不用于GEMM实现。没有依赖内核计算结果的加法链，也不会被编译器删除读操作。

cpufb 的 get_bandwith 使用外部频率换算B/cycle，本实验直接计数实际PMU周期。
连续load本身仍存在发射/端口限制，因此测的是该访问方式的可达吞吐，不是物理总线标称带宽。

拷贝用独立C++ NEON循环；每128字节源数据做8个向量load和8个store。
源和目标不重叠，确认阶段用memcmp验证输出。工作集大小为源+目标总量。
拷贝的有效带宽按读字节+写字节计算，不包括write-allocate和回写的额外总线流量。
不能据此将拷贝GB/s等同于cache层间实际传输GB/s。

## 计量和层级判断

- CLOCK_MONOTONIC_RAW测时间，GB/s用10^9字节/秒；KiB/MiB用1024进制。
- 五个PMU事件成组读取：cycles=0x11，L1D refill=0x03，L2D refill=0x17，
  LL read=0x36，LL read miss=0x37；均通过目标机sysfs事件文件核对编码。
- 排除kernel/hypervisor，固定group，要求time_enabled==time_running，禁止接受复用结果。
- B/cycle=实际循环有效字节数/PMU周期数；GHz=周期数/墙钟时间，用于检查频率与停顿。
- PMU refill/read/miss以每MiB有效字节的事件次数显示，未强行按cacheline换算总线流量。
  预取会影响demand refill，L1 refill不能直接解释成全部L1-L2流量。
- 目标sysfs：L1D64KiB/64B line、L2 512KiB/64B line、L3 32MiB/128B line。
  L1/L2私有，L3由CPU0-7共享。这些是容量，不是所有访问模式的驻留保证。
- 默认普通页，4096B对齐。目标THP为madvise，程序没有请求THP或显式大页。
  L3测量包含顺序地址转换开销，不把它声称为脱离TLB的纯物理L3端口峰值。

选择工作集必须结合PMU：L1驻留时refill接近0；L2候选需要超过L1、同时L2refill/LLread
接近0；L3候选需要明显LLread且LLmiss很低。不能简单用“容量的一半”给所有层定性。
这里测的是单核顺序流带宽，不是整机L3/内存聚合带宽，也不是指针追逐延迟。

## 本次确认结果

kp920，GCC13.2，单核约2.59–2.60GHz。9轮，每样本约250ms：

| 测试 | 工作集 | 纯读GB/s中位 | 九次min–max | B/cycle |
|---|---:|---:|---:|---:|
| L1驻留 | 32KiB | 81.423 | 81.406–81.439 | 31.350 |
| L2供数 | 128KiB | 54.009 | 53.891–54.076 | 20.801 |
| L3供数 | 4MiB | 15.484 | 14.123–19.213 | 5.971 |
| L3供数 | 8MiB | 14.628 | 13.495–15.817 | 5.638 |
| L3供数，ABC容量相当 | 12MiB | 14.351 | 13.630–14.950 | 5.532 |
| L3供数 | 16MiB | 14.400 | 13.788–14.656 | 5.551 |
| 超L3容量，内存为主 | 64MiB | 11.467 | 11.455–11.491 | 4.426 |

32KiB时L1 refill约0.08次/MiB，基本L1命中。
128KiB时L1 refill约3830次/MiB、L2 refill和LL read仅0.53次/MiB，基本没有L3流量。
12MiB时LL read约16384次/MiB、LL miss仅0.40次/MiB（事件比值约0.0024%），
数据主要来自L3。L3仍有明显波动，不能将14.351的小数位当成硬件固定精度。

相同工作集的拷贝有效读+写带宽：32KiB为67.623GB/s，128KiB为57.544GB/s，
12MiB为16.204GB/s。这是两条流（源和目标）的有效吞吐，不是额外测得的纯store峰值。

另一个重要观察：192/224/256KiB纯读已有约854/715/625次LL read每MiB，
所以不能用这些点充当纯L2带宽，更不能把之前特定GEMM负载下的224KiB预算
解释成所有访问方式都不会访问L3的保证。此处只记录现象，不凭本实验推断具体替换策略。

本次是普通页顺序循环负载，12MiB只是与ABC总容量相当，绝不是完整GEMM的访问过程。
用于估算GEMM时，还要结合panel复用、pack、C读写和load/FMLA同时执行的资源竞争。
