# ARM NEON Code Examples

Standalone C examples for learning ARM NEON intrinsics, AArch64 inline
assembly, CPU affinity, and matrix operations with I8MM and SME.

## Examples

| Source | Demonstration | Requirements |
| --- | --- | --- |
| [test1.c](test1.c) | Load one lane versus a full vector | ARM NEON |
| [test2.c](test2.c) | Unsigned 8-bit vector addition, subtraction, and multiplication | ARM NEON |
| [test3.c](test3.c) | Transpose a 2 x 2 double-precision matrix | AArch64 NEON |
| [test4.c](test4.c) | Transpose a 4 x 4 single-precision matrix | ARM NEON |
| [tset5.c](tset5.c) | Vector arithmetic using inline assembly | AArch64 NEON |
| [test6.c](test6.c) | Narrow 16-bit integers to 8-bit integers | ARM NEON |
| [test7.c](test7.c) | Count online CPUs and bind the process to CPU 0 | Linux CPU-affinity APIs |
| [test8.c](test8.c) | Compare scalar and NEON dot products | AArch64 NEON |
| [test9.c](test9.c) | Signed 8-bit matrix multiplication | AArch64 with I8MM |
| [test10.c](test10.c) | 8 x 8 floating-point matrix multiplication using ZA | SME-capable hardware, OS, and compiler |

The original filename `tset5.c` is retained intentionally.

## Build and run

Each source file has its own `main` function; compile examples separately.
Use GCC or Clang on an ARM64 machine. Ordinary NEON examples can run on
ARM64 Linux and Apple Silicon macOS:

```sh
mkdir -p build
cc -O2 -Wall -Wextra test1.c -o build/test1
./build/test1
```

Replace `test1.c` with the desired source. For the dot-product example:

```sh
cc -O2 -Wall -Wextra test8.c -lm -o build/test8
./build/test8
```

CPU affinity is Linux-specific and cannot be built unchanged on macOS:

```sh
cc -O2 -Wall -Wextra test7.c -o build/test7
./build/test7
```

CPU 0 must be permitted by the process's affinity restrictions.

I8MM requires a suitable target flag and a CPU implementing the extension:

```sh
cc -O2 -Wall -Wextra -march=armv8.6-a+i8mm test9.c -o build/test9
./build/test9
```

SME requires a compiler providing `<arm_sme.h>` and the SME ACLE attributes
and intrinsics, as well as hardware and OS support:

```sh
clang -O2 -Wall -Wextra -march=armv9-a+sme test10.c -o build/test10
./build/test10
```

Compilation alone does not establish hardware support. Do not run the I8MM
or SME examples on CPUs that lack the required extension.

## Recorded results

The `*-output.txt` files contain previously recorded example output on
Huawei Kunpeng 920 and NVIDIA Jetson Nano. The
[test7_kunpeng920_output.txt](test7_kunpeng920_output.txt) file records CPU
affinity output; [test10_local_output.txt](test10_local_output.txt) records
a local SME matrix-multiplication result.

[likwid-topology-kp920.txt](likwid-topology-kp920.txt) and
[likwid-topology-nvidia-nano.txt](likwid-topology-nvidia-nano.txt) contain
historical LIKWID hardware-topology output. Recorded output is reference
material, not a guarantee that every example runs on every listed device.

## Reference materials

- [arm-neon.pdf](arm-neon.pdf): ARM NEON reference document.
- [ARM NEON Programing.pptx](ARM%20NEON%20Programing.pptx): ARM NEON
  presentation.
- [env.md](env.md): Sanitized cluster environment and job-submission notes.
- [agent.md](agent.md): Sanitized benchmark-device access templates.

Private keys, unsanitized internal device-access records, device identifiers,
and local build products are excluded from version control.

## Licensing

This repository is public, but no open-source license has been granted yet.
Public visibility alone does not grant permission to reuse, modify, or
redistribute its contents.
