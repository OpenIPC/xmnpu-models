# sanitize.py <a.xmm> <b.xmm> <out.xmm>
#
# XMTVMC writes host memory into the tensor table at the end of an .xmm: whole
# 8-byte host pointers, and 4-byte struct padding left uninitialised beside a
# 32-bit field, which holds the stale upper half of one. Neither means
# anything on the camera, but both follow the build host's memory layout, so
# they make every build different.
#
# <a> and <b> are the same model built twice with different address layouts
# (compile.sh builds <b> with address-space randomisation on). Two passes,
# over 4-byte words, written to <out> from <a>:
#
#   1. every word that differs between the builds is zeroed -- that word
#      only: the 32-bit value beside a word of padding is real and the same in
#      both, and stays;
#   2. from the first difference to the end of the file -- the tensor table --
#      every word whose value is the upper half of an x86-64 user-space stack
#      (0x7ff0-0x7fff) or heap (0x5500-0x56ff) address is zeroed too. Under
#      randomisation a stack address's upper half takes only a few values, so
#      the two builds often agree on one and pass 1 cannot see it; the v4
#      release kept three that way. The same values occur in weights, which is
#      why this pass stops at the table.
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

differing = {i - i % 4 for i in diff}
table = diff[0] - diff[0] % 4
upper_half = {w for w in range(table, len(a) - 3, 4)
              if w not in differing and
              (0x7ff0 <= struct.unpack_from("<I", a, w)[0] <= 0x7fff or
               0x5500 <= struct.unpack_from("<I", a, w)[0] <= 0x56ff)}
words = differing | upper_half
if len(words) > MAX_WORDS:
    sys.exit(f"sanitize: {len(words)} words to zero, more than host memory explains")
for w in words:
    a[w:w + 4] = bytes(4)
open(sys.argv[3], "wb").write(a)
print(f"sanitize: zeroed {len(differing)} differing words and "
      f"{len(upper_half)} coincident pointer halves")
