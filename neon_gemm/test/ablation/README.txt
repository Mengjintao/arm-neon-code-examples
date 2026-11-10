Run on kp920 from gemm_neon: bash test/ablation/run.sh
Output: result.out / error.err (overwritten). Temporary .ablation is removed.
Requires GCC AArch64, Python3, objdump, taskset.

prepare.py extracts the K body from step/register_lane.cpp and adapts B's
stride to the shared packed-panel layout. It does not compile or modify the
step file's original driver. All variants use src/main.cpp's GEMM driver,
MC/NC/KC from gemm.h, and the same initial-C behavior. Main allocation,
random initialization and reference checking are outside e2e timing.

V0: register-block K loop, scalar A pack, no software prefetch.
V1: duplicate K body twice, same pack and no software prefetch.
V2: src two-step pipeline, no software prefetch.
V3: V2 plus next-A, next-B and caller's C prefetch (current scalar version).
V4: V3 with pack_v_buffer.cpp instead of pack_s.cpp.
B pack differs too (scalar vs NEON double buffer). A pack implementations
differ in grouping as well as intrinsics; V4 is an implementation comparison, not an isolated
vectorization compiler switch. Compiler auto-vectorization stays enabled.

Deletion variants use V4 as reference. drop_pipeline replaces the pipelined
body with an unrolled sequential load/FMLA body, retaining A/B prefetch
coverage. Prefetch scheduling changes with the body; its result is not a
perfect instruction-for-instruction scheduling ablation.
plain_unroll4 compares with V1.

All kernels compiled with identical -O3 -mcpu=tsv110; scheduling enabled.
Shared libraries bind their own symbols with -Bsymbolic; all versions use
identical allocation addresses and the same harness. Absolute timing may
differ from a separately linked executable. Interpret within-run deltas.

CPU0, 3 rounds of 15 randomized interleaved samples per variant, each with
2 untimed warmups. Report each round's median, then median of three medians.
Full 1024^3 and remainder cases checked against a double reference.
codegen.py prints backward arithmetic-loop FMLA counts, instruction counts,
vector stack access counts and prefetch counts. Nested ranges may overlap;
these are static code checks, not measured dynamic instruction counts.

The sequential unroll2+prefetch variant groups two B hints at the start of
each two-K iteration, like the reference pipeline. A preliminary per-K
hint formulation let GCC move B hints outside the null-pointer guard and
was extremely slow; that preliminary run is discarded. The grouped version
was checked for a preserved B-null dispatch before rerunning all variants.
