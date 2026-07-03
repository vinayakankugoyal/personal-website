---
layout: doc
---

# Pataka: Booting Linux in a VM in Long Mode

*July 2, 2026*

---

Pataka means firecracker in Hindi, and that's what this is: a minimal Virtual Machine Monitor inspired by [Firecracker](https://github.com/firecracker-microvm/firecracker). In this post I want to boot a real 64-bit Linux kernel inside a VM, which means getting the guest CPU into long mode.

A virtual CPU, just like a real x86 CPU, powers on in 16-bit real mode, but a modern Linux kernel is 64-bit code. Nobody bridges that gap for us. The VMM has to build every structure the CPU needs to get from real mode to long mode (page tables, a GDT, the right control register bits) before the kernel's first instruction runs. Get one bit wrong and the guest triple faults[^1] before it prints a single character.

## Table of Contents

[[toc]]

## Pre-requisite

I'm using NixOS, so here's the `shell.nix`:

```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustc
    cargo
    rustfmt
    clippy
    rust-analyzer
    rustPlatform.rustLibSrc
    pkg-config
  ];

  RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
}
```

::: info 📝 Note
If you're using VS Code with Nix, launch it from within the nix-shell so rust-analyzer can find the toolchain:
```bash
cd ~/projects/pataka && nix-shell --run "code ."
```
:::

## The three CPU modes (and why we care)

An x86 CPU goes through three modes on its way to running 64-bit code:

1. **Real mode (16-bit).** Where the CPU starts. 20-bit segmented addressing, no paging, no memory protection. This is where our virtual CPU wakes up.
2. **Protected mode (32-bit).** Adds segmentation with privilege levels and (optionally) paging.
3. **Long mode (64-bit).** What the kernel wants. Flat 64-bit addressing, paging is mandatory, segments are mostly ignored.

There is no "be 64-bit now" switch. The CPU refuses to enter long mode unless paging is enabled, paging needs valid page tables in memory, and enabling paging needs a valid GDT already loaded. Our job is to set up all of those preconditions in guest memory and then set the control register bits, so that by the time the kernel's entry point runs the CPU is already a 64-bit machine.

## KVM in one paragraph

KVM (Kernel-based Virtual Machine) is a Linux kernel module that turns the kernel into a hypervisor. It exposes hardware virtualization (Intel VT-x / AMD-V) through `/dev/kvm`. KVM lets you create an isolated virtual machine and run guest code directly on the physical CPU. This isn't emulation, it's near-native speed. What KVM does *not* do is know what a disk is, what a network card is, or what a screen is. It just runs CPU instructions and manages memory isolation. Everything else, including all the mode-transition setup below, is the VMM's job in plain userspace.

## Bringing up a VM and a vCPU

The plumbing is short. KVM works through a hierarchy of file descriptors: a system fd from `open("/dev/kvm")`, a VM fd for one virtual machine, and a vCPU fd for one virtual processor.

```bash
cargo init --name pataka
cargo add kvm-ioctls kvm-bindings libc elf
```

```rust
let kvm = Kvm::new()?;
let vm_fd = kvm.create_vm()?;
```

The guest needs RAM, so we `mmap` a chunk of memory and tell KVM to use it as the guest's physical address space:

```rust
let addr = unsafe {
    libc::mmap(
        std::ptr::null_mut(),
        256 * 1024 * 1024,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    )
};

let kvm_memory = kvm_userspace_memory_region {
    slot: 0,
    guest_phys_addr: 0,
    memory_size: 256 * 1024 * 1024,
    userspace_addr: addr as u64,
    flags: 0,
};

unsafe { vm_fd.set_user_memory_region(kvm_memory)?; }
```

From the guest's perspective, it has 256MB of RAM starting at address `0x0`. From our perspective, it's just a chunk of our own process memory. When the guest reads address `0x1000`, KVM translates that to the corresponding offset in our `mmap`'d region.

```rust
let vcpu_fd = vm_fd.create_vcpu(0)?;
```

That fresh vCPU is sitting in 16-bit real mode right now. The rest of this post is about what we write into memory and its registers before we ever call `run()`.

## Loading the kernel

I grabbed a prebuilt Firecracker test kernel (`vmlinux.bin`), which is an ELF binary. We parse the ELF headers, find the `PT_LOAD` segments, and copy each one into guest memory at the physical address the ELF specifies. The ELF entry point tells us where the 64-bit code begins, which is the address we'll eventually set `rip` to. The kernel loads at `0x1000000` (16 MB).

The important thing: this is 64-bit code. The moment the CPU jumps to that entry point it must already be in long mode, or it will decode those bytes as garbage and triple fault.

## The guest memory layout

Here is everything we are going to write into guest physical memory: the GDT and page tables for the mode transition, and the boot params for the kernel to read once it's running.

```
Guest Physical Memory (256 MB)
┌─────────────────────────────────────┐ 0x0
│                                     │
│  (unused low memory)                │
│                                     │
├─────────────────────────────────────┤ 0x1000
│  GDT (4 entries × 8 bytes)          │ ← sregs.gdt.base
├─────────────────────────────────────┤ 0x1020
│  IDT (empty)                        │ ← sregs.idt.base
├─────────────────────────────────────┤
│                                     │
├─────────────────────────────────────┤ 0x10000
│  Boot Parameters (4096 bytes)       │ ← rsi points here
│  ├─ 0x1E8: e820 entry count         │
│  ├─ 0x1FE: boot flag (0xAA55)       │
│  ├─ 0x202: header magic ("HdrS")    │
│  ├─ 0x210: type_of_loader           │
│  ├─ 0x228: cmd_line_ptr → 0x50000   │
│  └─ 0x2D0: e820 memory map          │
├─────────────────────────────────────┤ 0x1fff0
│  Stack (grows downward)             │ ← rsp, rbp
├─────────────────────────────────────┤
│                                     │
├─────────────────────────────────────┤ 0x40000
│  PML4 (page table level 4)          │ ← cr3 points here
├─────────────────────────────────────┤ 0x41000
│  PDPTE (page table level 3)         │
├─────────────────────────────────────┤ 0x42000
│  PDE (512 entries × 8 bytes)        │ 512 × 2MB = 1GB identity mapped
├─────────────────────────────────────┤
│                                     │
├─────────────────────────────────────┤ 0x50000
│  Kernel command line                │ "console=ttyS0 ..."
├─────────────────────────────────────┤
│                                     │
├─────────────────────────────────────┤ 0x1000000 (16 MB)
│                                     │
│  Kernel (ELF PT_LOAD segments)      │ ← rip starts here
│  .text, .data, .bss, etc.           │
│  (~30 MB)                           │
│                                     │
├─────────────────────────────────────┤ ~0x2000000
│                                     │
│  Free RAM                           │
│                                     │
└─────────────────────────────────────┘ 0x10000000 (256 MB)
```

None of these addresses are special. Firecracker puts its GDT at `0x500` and its page tables at `0x9000`; I picked different spots on purpose to show the CPU doesn't care. The only rules: page tables must be 4KB aligned and nothing can overlap. `cr3`, `gdt.base`, and `rsi` tell the CPU and the kernel where everything is.

The next three sections cover the page tables, the GDT, and the control register writes that flip the CPU into long mode.

## Page tables

The CPU cannot be in long mode without paging, and paging needs valid page tables that `cr3` points at. So we build the tables first.

In long mode, address translation is a 4-level hierarchy, but we take a shortcut: we identity map the first 1GB using 2MB pages, which lets us stop after 3 levels. The CPU walks PML4 (Page Map Level 4) → PDPTE (Page Directory Pointer Table Entry) → PDE (Page Directory Entry) on every memory access, and because we set the "page size" bit in the PDE entries, it stops there instead of going down to a 4th level of 4KB pages.

```
Virtual Address (48 bits used)
┌───────────┬────────────┬────────────┬─────────────────────┐
│ PML4 idx  │ PDPTE idx  │ PDE idx    │ Offset within 2MB   │
│ bits 47-39│ bits 38-30 │ bits 29-21 │ bits 20-0           │
└────┬──────┴────┬───────┴────┬───────┴─────────────────────┘
     │           │            │
     ▼           │            │
  PML4 @ 0x40000 │            │          We only have 1 entry:
  ┌──────────┐   │            │          PML4[0] → PDPTE @ 0x41000
  │ entry 0  │   │            │
  │ ...      │   │            │
  └──────────┘   │            │
                 │            │
     ▼───────────┘            │
  PDPTE @ 0x41000             │          We only have 1 entry:
  ┌──────────┐                │          PDPTE[0] → PDE @ 0x42000
  │ entry 0  │                │
  │ ...      │                │
  └──────────┘                │
       ▼──────────────────────┘
  PDE @ 0x42000                      512 entries, each maps 2MB:
  ┌──────────┐
  │ entry 0  │ → 0x00000000 - 0x001FFFFF  (2 MB)
  │ entry 1  │ → 0x00200000 - 0x003FFFFF  (2 MB)
  │ entry 2  │ → 0x00400000 - 0x005FFFFF  (2 MB)
  │ ...      │
  │ entry 511│ → 0x3FE00000 - 0x3FFFFFFF  (2 MB)
  └──────────┘
                 512 × 2MB = 1GB identity mapped
```

Let's trace an address through the tables. Take `0x0000000001034567`, an address inside the kernel:

```
virtual address 0x0000000001034567, split at the field boundaries in binary:

0000000000000000 000000000 000000000 000001000 000110100010101100111
│    63-48     │ │ 47-39 │ │ 38-30 │ │ 29-21 │ │       20-0        │
│    unused    │ │ PML4  │ │ PDPTE │ │  PDE  │ │      offset       │
│     = 0      │ │  = 0  │ │  = 0  │ │  = 8  │ │     = 0x34567     │

cr3 = 0x40000                                 (where every walk starts)
 └─ PML4[0]  @ 0x40000 = 0x41003              → PDPTE is at 0x41000
     └─ PDPTE[0] @ 0x41000 = 0x42003          → PDE is at 0x42000
         └─ PDE[8]  @ 0x42040 = 0x1000083     → PS bit set: 2MB page at 0x1000000
             └─ 0x1000000 + 0x34567 = physical 0x1034567
```

The fields don't line up with hex digits: the 9-bit indexes put boundaries at bits 21, 30, and 39, none of which are multiples of 4. The `1` in the hex is bit 24, which lands inside the PDE index field, and those 9 bits (`000001000`) read as 8. The low bits of each entry (`0x3`, `0x83`) are flags: present, writable, and in the PDE's case the page size bit that means "this entry maps a 2MB page directly".

Identity mapping means virtual address `0x1000000` maps to physical address `0x1000000`: the walk ends at the same number it started with. That's what we want, because the kernel loads at physical `0x1000000` and `rip` will point at that address. The kernel builds its own page tables later; ours only have to carry it through early boot.

::: tip 💡 Intuition: Mapping is not allocating
Why map 1GB when the guest only has 256MB of RAM? Because a page table entry costs 8 bytes, not 2MB. It's just a translation rule: "if you see virtual address X, send it to physical X". The kernel learns what RAM exists from the e820 map, not from the page tables, so the space between 256MB and 1GB is never touched. We fill all 512 PDE entries because one full table is less code than computing the exact count; 128 entries would boot identically.
:::

Why do addresses only use 48 of their 64 bits? Because the width falls out of the table geometry. Each table is one 4KB page holding 512 entries, so each level translates 9 bits, and the offset within a 4KB page takes 12: four levels gives 9 × 4 + 12 = 48. Translating all 64 bits would need six levels, two extra memory reads on every table walk, for memory that won't exist for decades. So the hardware stops at 48 bits (256 TB) and requires bits 63 to 48 to be copies of bit 47, a "canonical" address. That rule stops software from hiding data in the unused bits, which is what let CPUs later add a 5th level (57 bits) without breaking anything.

Nothing stops you from going the full 4 levels with 4KB pages either: clear the page size bit and each PDE entry points at a page table whose 512 entries map 4KB apiece. The mapping would behave identically, but the same 1GB would take 512 page table pages (2MB of tables) instead of one, which is why 2MB pages are the standard choice for early boot.

## The GDT

Before the CPU will enter protected mode, it needs a valid Global Descriptor Table. The GDT defines memory segments, so first: what is segmentation? It's x86's original memory protection scheme, older than paging. Every memory access goes through a segment: a base address, a limit, and permission bits packed into an 8-byte descriptor. An access like `ds:0x1234` means "offset `0x1234` from the base of the data segment", and the CPU faults if the offset runs past the limit or the permissions don't allow it. The GDT holds those descriptors, and the segment registers (`cs`, `ds`, `ss`, ...) index into it. We write 4 entries (null, 64-bit code, data, TSS) at `0x1000` and point all the segment registers at them.

In 64-bit mode this is mostly a formality because long mode ignores segment bases and limits. But the CPU refuses to run without a well-formed GDT, and the code segment descriptor carries the `L` (long) bit that marks the segment as 64-bit code. So we set it up carefully even though it does almost nothing at runtime.

## The control registers

With page tables at `0x40000` and a GDT at `0x1000`, we can set the control registers. Long mode only turns on when a specific combination of bits is correct all at once:

- **`cr3`** points at the PML4 at `0x40000`, so the CPU knows where the page tables live.
- **`cr4`** enables PAE (Physical Address Extension). Long mode requires PAE-style page tables, so this has to be on before paging.
- **EFER** gets the LME bit (Long Mode Enable) set. This *arms* long mode but doesn't activate it yet.
- **`cr0`** enables protected mode (PE bit) and paging (PG bit). The instant paging turns on while LME is armed and PAE is set, the CPU latches into long mode. This is the actual transition.

With KVM we don't need a hand-written assembly trampoline. We set the special registers directly through `set_sregs`, so the vCPU starts in long mode from its very first instruction:

```rust
let mut sregs = vcpu_fd.get_sregs()?;

// point at our page tables and GDT
sregs.cr3 = 0x40000;
sregs.gdt.base = 0x1000;
sregs.gdt.limit = 4 * 8 - 1;

// PAE on
sregs.cr4 = CR4_PAE;
// long mode enable + long mode active
sregs.efer = EFER_LME | EFER_LMA;
// protected mode + paging  → CPU is now in long mode
sregs.cr0 = CR0_PE | CR0_PG;

// 64-bit code segment with the L bit set
sregs.cs = long_mode_code_segment();
sregs.ds = data_segment();
// ... ss, es, fs, gs likewise

vcpu_fd.set_sregs(&sregs)?;
```

Then the general registers point the CPU at the kernel:

```rust
let mut regs = vcpu_fd.get_regs()?;
regs.rip = kernel_entry;   // 64-bit kernel entry point
regs.rsi = 0x10000;        // boot_params ("zero page")
regs.rsp = 0x1fff0;        // stack, grows down
regs.rflags = 2;           // bit 1 must always be set
vcpu_fd.set_regs(&regs)?;
```

The stack grows downward on x86, so `rsp = 0x1fff0` is the top: every push moves it toward lower addresses, toward the boot params at `0x10000` and away from the page tables above it at `0x40000`.

That `rsi = 0x10000` is the [Linux/x86 boot protocol](https://docs.kernel.org/arch/x86/boot.html) calling convention: the kernel expects a pointer to its `boot_params` in `rsi` on entry. That document is the source of truth for everything in this section, including the register state the kernel wants at the 64-bit entry point and every field in the zero page.

## Boot parameters: what the kernel reads on arrival

The CPU is in long mode and `rip` points at the kernel, but the kernel still needs to be told about its environment. It expects a 4096-byte `boot_params` struct (the "zero page") at `0x10000`. The two fields that matter most are the command line pointer (`console=ttyS0` so kernel output goes to the serial port where we can see it) and the e820 memory map.

The e820 map tells the kernel which physical memory regions exist and what each is for. The name comes from BIOS interrupt `0xE820`, which is how real hardware reports its memory layout. Since there's no BIOS here, we synthesize it:

```
e820 Memory Map (3 entries)
┌──────────────────────────────────────────────────────────────┐
│ Entry 0:  0x00000000 - 0x0009FBFF  (639 KB)    type=usable   │  conventional low memory
│ Entry 1:  0x0009FC00 - 0x000FFFFF  (385 KB)    type=reserved │  EBDA + legacy ROMs
│ Entry 2:  0x00100000 - 0x0FFFFFFF  (~255 MB)   type=usable   │  main RAM for the kernel
└──────────────────────────────────────────────────────────────┘
```

On real hardware there would be many more entries (holes for PCI devices, ACPI tables, and so on), but for our VMM these three are enough. The kernel uses this map to figure out where it can safely allocate memory.

## Two gotchas that cost me hours

Even with the CPU correctly in long mode, two things silently broke the boot:

**CPUID.** The kernel runs `cpuid` instructions very early to discover CPU features. Without calling `KVM_SET_CPUID2`, the guest gets garbage results and triple faults right after entry. The fix is one line: pass the host's supported CPUID through to the guest.

**PIT and IRQ chip.** The kernel calibrates timers using the PIT (Programmable Interval Timer). Rather than emulating that in userspace, KVM can handle it in-kernel via `create_irq_chip()` and `create_pit2()`.

Both of these are technically separate from the mode transition, but they're the difference between "boots into long mode and then triple faults" and "boots into long mode and keeps going."

## The run loop

Once the vCPU is set up and running in long mode, the VMM's job shrinks to a single loop. Every time the guest does I/O, KVM pauses the vCPU and returns control to us:

```
loop {
    tell KVM to run the vCPU
    KVM runs guest code on real hardware in long mode...
    ...guest does I/O (like writing to the serial port)...
    KVM exits back to you with: "guest wrote byte 'H' to port 0x3f8"
    you handle it
}
```

## It boots!

```
[    0.000000] Linux version 4.14.55-84.37.amzn2.x86_64 ...
[    0.000000] Command line: console=ttyS0 earlyprintk=serial,ttyS0,115200
[    0.000000] e820: BIOS-provided physical RAM map:
[    0.000000] BIOS-e820: [mem 0x0000000000000000-0x000000000009fbff] usable
[    0.000000] BIOS-e820: [mem 0x000000000009fc00-0x00000000000fffff] reserved
[    0.000000] BIOS-e820: [mem 0x0000000000100000-0x000000000fffffff] usable
[    0.000000] NX (Execute Disable) protection: active
...
[    0.312000] Freeing unused kernel memory: 1024K
[    0.318000] VFS: Cannot open root device "(null)" or unknown-block(0,0): error -6
[    0.318000] Please append a correct "root=" boot option; here are the available partitions:
[    0.319000] Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)
[    0.319000] CPU: 0 PID: 1 Comm: swapper/0 Not tainted 4.14.55-84.37.amzn2.x86_64 #1
[    0.319000] Call Trace:
[    0.319000]  dump_stack+0x5c/0x82
[    0.319000]  panic+0xe4/0x252
[    0.319000]  mount_block_root+0x2b1/0x2c0
[    0.319000]  ---[ end Kernel panic - not syncing: VFS: Unable to mount root fs ... ]---
```

Those messages are a 64-bit kernel running in long mode inside our VM. It boots, initializes subsystems, and then hits that final panic. It's expected, because we haven't given it a root filesystem yet, so once the kernel finishes early init it has nothing to mount and PID 1 never starts. The mode transition worked; the next step is an initramfs or virtio-block.

## What I learned

1. **Long mode is a series of preconditions, not a switch.** The CPU only enters 64-bit mode when PAE, valid page tables, long mode enable, and paging are all set correctly at once. Miss any one and you triple fault.
2. **You build the boot environment by hand.** Real hardware has firmware and a bootloader to set up page tables and the GDT. In a VMM, that's all you, written straight into guest memory.
3. **x86 carries its whole history.** Real mode, protected mode, long mode, GDT, page tables, boot params. Decades of backwards compatibility baked into silicon, and you walk through all of it just to reach the kernel's first instruction.
4. **Small details triple fault you.** Forgetting CPUID setup caused a triple fault. Returning the wrong value for the serial status register caused a 100x slowdown.
5. **KVM itself is simple.** It's just a handful of ioctls. All the complexity lives in the memory layout and the mode transition around it.

If you liked this post please share it with your friends!

You can find the complete implementation [here](https://github.com/vinayakankugoyal/pataka).

---

[^1]: A triple fault is what happens when the CPU fails to recover from an error three times over. When the CPU hits a problem it raises a *fault* and jumps to a handler you registered in the IDT. If handling that fault causes another fault, that's a *double fault*, and the CPU tries a dedicated double-fault handler. If handling that faults too, the CPU gives up: that's the triple fault. On real hardware the machine resets. Under KVM the vCPU just exits with a shutdown. During early boot there are no fault handlers installed yet, so the very first mistake cascades straight to a triple fault, which is why a single wrong bit gets you a silent dead machine instead of an error message.
