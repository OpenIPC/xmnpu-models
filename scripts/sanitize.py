# sanitize.py <a.xmm> <b.xmm> <out.xmm>
#
# XMTVMC writes host memory into the tensor table at the end of an .xmm: whole
# 8-byte host pointers, and 4-byte struct padding left uninitialised beside a
# 32-bit field, which holds the stale upper half of one. Neither means
# anything on the camera, but both follow the build host's memory layout, so
# they make every build different.
#
# <a> and <b> are the same model built twice: <a> without address-space
# randomisation, <b> (compile.sh) with it. Every 4-byte word that differs
# between them is host memory, and is zeroed -- that word only: the 32-bit
# value beside a word of padding is real, the same in both, and stays.
#
# Except a word that holds the upper half of a host address. Under
# randomisation that half takes only a few values, so on some builds <b>
# agrees with <a> on it and it does not differ at all; zeroing it only when it
# happens to differ would make the output depend on <b>'s luck, which is how
# the v4 release kept three that a rebuild of its commit zeroed. Without
# randomisation it is fixed on any x86-64 host -- 0x7fff for the stack and
# mmap, 0x5555 for the PIE heap -- so such a word keeps <a>'s value whether or
# not it differs (whatever <b> holds there), and the output is the same every
# time. A word that is equal in the two builds is never changed.
#
# Refuses, rather than zeroing anything else, when the builds differ in a way
# host memory cannot explain: different sizes, a run of differing bytes longer
# than a pointer, or more than MAX_WORDS words.
import struct
import sys

MAX_WORDS = 512

a = bytearray(open(sys.argv[1], "rb").read())
b = open(sys.argv[2], "rb").read()
if len(a) != len(b):
    sys.exit(f"sanitize: builds differ in size ({len(a)} and {len(b)})")

diff = [i for i in range(len(a)) if a[i] != b[i]]
if not diff:
    sys.exit("sanitize: the two builds are identical; the second was not randomised")
run_lo = run_hi = diff[0]
for i in diff[1:] + [None]:
    if i is not None and i == run_hi + 1:
        run_hi = i
        continue
    if run_hi - run_lo >= 8:
        sys.exit(f"sanitize: {run_hi - run_lo + 1} differing bytes at {run_lo:#x}, "
                 "longer than a pointer")
    if i is not None:
        run_lo = run_hi = i

UPPER_A = (0x7FFF, 0x5555)          # the halves without randomisation


def word(buf, w):
    return struct.unpack_from("<I", buf, w)[0]


differing = sorted({i - i % 4 for i in diff})
kept = [w for w in differing if word(a, w) in UPPER_A]
zeroed = [w for w in differing if w not in set(kept)]
if len(zeroed) > MAX_WORDS:
    sys.exit(f"sanitize: {len(zeroed)} words to zero, more than host memory explains")
for w in zeroed:
    a[w:w + 4] = bytes(4)
open(sys.argv[3], "wb").write(a)
print(f"sanitize: zeroed {len(zeroed)} differing words; kept {len(kept)} "
      "address upper halves at their unrandomised value")
