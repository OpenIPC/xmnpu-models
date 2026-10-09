# sanitize.py <a.xmm> <b.xmm> <out.xmm>
#
# XMTVMC copies some of its own host pointers into the .xmm -- 8-byte ones
# and the low halves of others -- in the tensor table at the end of the file.
# They mean nothing on the camera (zeroed, the model is still bit-exact with
# its C-model golden on the NPU) but they make every build different, since
# they follow the build host's memory layout.
#
# <a> and <b> are the same model built twice with different address layouts
# (compile.sh builds <b> with address-space randomisation on). Every byte that
# differs is part of such a pointer: each run of them is widened to its whole
# field -- 8 bytes when that holds a user-space x86-64 pointer in both files,
# otherwise the aligned 32-bit words -- and zeroed in <a>, written to <out>.
#
# Refuses, rather than zeroing anything else, when the builds differ in a way
# a pointer cannot explain: different sizes, a run longer than a pointer, or
# more than MAX_FIELDS fields.
import struct
import sys

MAX_FIELDS = 64

a = bytearray(open(sys.argv[1], "rb").read())
b = open(sys.argv[2], "rb").read()
if len(a) != len(b):
    sys.exit(f"sanitize: builds differ in size ({len(a)} and {len(b)})")


def user_ptr(buf, off):
    if off < 0 or off + 8 > len(buf):
        return False
    v = struct.unpack_from("<Q", buf, off)[0]
    return 0x1000 <= v < 0x800000000000 and v >> 40 != 0


diff = [i for i in range(len(a)) if a[i] != b[i]]
runs = []
for i in diff:
    if runs and i <= runs[-1][1] + 1:
        runs[-1][1] = i
    else:
        runs.append([i, i])
fields = set()
for lo, hi in runs:
    if hi - lo >= 8:
        sys.exit(f"sanitize: {hi - lo + 1} differing bytes at {lo:#x}, longer than a pointer")
    starts = [s for s in range(hi - 7, lo + 1)
              if s % 4 == 0 and user_ptr(a, s) and user_ptr(b, s)]
    if starts:
        fields.add((starts[0], 8))
    else:
        for w in range(lo - lo % 4, hi + 1, 4):
            fields.add((w, 4))
if len(fields) > MAX_FIELDS:
    sys.exit(f"sanitize: {len(fields)} differing fields, more than pointers explain")
for off, size in fields:
    a[off:off + size] = bytes(size)
open(sys.argv[3], "wb").write(a)
print(f"sanitize: zeroed {len(fields)} host-pointer fields "
      f"({sum(s for _, s in fields)} bytes)")
