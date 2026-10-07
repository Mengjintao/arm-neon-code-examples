# Benchmark Device Access (Sanitized)

This public version describes access workflows for ARM benchmark devices.
Actual server addresses, account names, SSH aliases, device serial numbers,
host-key fingerprints, internal paths, and historical login results have
been removed. The original access notes are backed up locally outside this
repository.

Never store passwords, access tokens, or private-key contents in this
repository. Keep SSH configuration and private keys outside the repository,
and verify server host keys through a trusted channel before connecting.
Do not disable strict host-key checking or replace a changed host key
without verification.

## Placeholder configuration

The commands below use environment variables that must be configured
locally with values supplied by your device or cluster administrator:

| Variable | Meaning |
| --- | --- |
| `JUMP_HOST` | Your local SSH alias for an authorized jump host |
| `CLUSTER_LOGIN` | Your local SSH alias for the cluster login node |
| `CLUSTER_QUEUE` | A queue available to your cluster account |
| `KUNPENG_HOST` | Your local SSH alias for the Kunpeng device |
| `NANO_HOST` | Your local SSH alias for the NVIDIA Nano |
| `XEON_HOST` | Your local SSH alias for the Xeon device |
| `RISCV_HOST` | Your local SSH alias for the RISC-V device |
| `PHONE_SERIAL` | The serial selected from your local `adb devices -l` output |
| `CPUFB_DIR` | Your local or remote cpufb project directory |

Configure authentication, usernames, ports, and any jump routes in your
local SSH configuration. These commands are templates, not verified
connections to public services.

## Huawei Kunpeng 920F

Connect to the cluster login node through an authorized jump host:

```bash
ssh -J "${JUMP_HOST:?Set JUMP_HOST}" "${CLUSTER_LOGIN:?Set CLUSTER_LOGIN}"
```

Do not run benchmarks on a login node. Query the available queues and
request an interactive compute-node allocation:

```bash
dqueue
dsub -I -q "${CLUSTER_QUEUE:?Set CLUSTER_QUEUE}" bash
```

Run benchmarks only after the allocation has entered a compute node.
For long benchmarks, prefer a batch job so an SSH disconnect cannot
truncate the output. See [env.md](env.md) for sanitized scheduling examples.

## Huawei Kunpeng 920

Use the locally configured SSH alias, with an authorized jump host when
required by the network:

```bash
ssh -J "${JUMP_HOST:?Set JUMP_HOST}" "${KUNPENG_HOST:?Set KUNPENG_HOST}"
```

If your administrator permits direct access from the current network:

```bash
ssh "${KUNPENG_HOST:?Set KUNPENG_HOST}"
```

## NVIDIA Nano

Choose a direct connection or an authorized jump route according to your
network. Store the appropriate port and identity-file configuration in
your local SSH configuration:

```bash
ssh "${NANO_HOST:?Set NANO_HOST}"
ssh -J "${JUMP_HOST:?Set JUMP_HOST}" "${NANO_HOST:?Set NANO_HOST}"
```

## Intel Xeon

```bash
ssh "${XEON_HOST:?Set XEON_HOST}"
```

If a server host key has changed, stop and verify it through a trusted
channel before reconnecting. Hardware performance counters may require
elevated privileges. Where authorized, run the benchmark interactively
and enter any sudo password directly in the terminal:

```bash
cd "${CPUFB_DIR:?Set CPUFB_DIR}"
sudo taskset -c 0 build/native-release/cpufb \
  '--thread_pool=[0]' --mode=all
```

CPU 0 must be permitted by the process's affinity restrictions.

## Android phones

These workflows apply to Xiaomi 11, Xiaomi Mi 8 SE, and OnePlus devices.
Connect the phone by USB, enable USB debugging, and authorize the computer.
On the computer to which the phone is attached:

```bash
adb devices -l
adb -s "${PHONE_SERIAL:?Set PHONE_SERIAL}" shell
```

If the phone is attached to an authorized remote host, connect to that
host first and run the ADB commands there. Select the correct serial on
that host rather than assuming a previously recorded identifier.

For a cpufb executable deployed to `/data/local/tmp/cpufb`, Android
`taskset` accepts a hexadecimal mask without a `0x` prefix. If core 7
exists and is permitted, its mask is `80`:

```bash
adb -s "${PHONE_SERIAL:?Set PHONE_SERIAL}" shell \
  'taskset 80 /data/local/tmp/cpufb --thread_pool=[7] --mode=all'
```

## RISC-V device

Use a network and SSH route authorized for the device:

```bash
ssh "${RISCV_HOST:?Set RISCV_HOST}"
```

On the device, select the project directory and run a compatible build:

```bash
cd "${CPUFB_DIR:?Set CPUFB_DIR}"
./cpufb '--thread_pool=[0]' --mode=all
```

## Apple Silicon

No remote connection is needed for a local Apple Silicon machine.
The following commands refer to a separate cpufb checkout with the
`native-release` CMake presets; they are not build commands for this
repository's standalone C examples:

```bash
cd "${CPUFB_DIR:?Set CPUFB_DIR}"
cmake --preset native-release
cmake --build --preset native-release
./build/native-release/cpufb '--thread_pool=[0]' --mode=all
```

See [README.md](README.md) for this repository's example build commands.
