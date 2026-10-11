# xmtvmc-env.sh <wheels>, sourced: XMTVMC's Python 3.8 environment in an
# ubuntu:20.04 container, as compile.sh and simulate.sh both need it. Sets
# $venv, $sp (its site-packages), LD_LIBRARY_PATH and PATH.
wheels=$(readlink -f "$1")
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3.8 python3.8-venv \
	libpython3.8 libhdf5-103 libtinfo5 libgl1 libglib2.0-0 libgomp1 \
	g++ patch unzip ca-certificates >/dev/null

venv=/opt/xmtvmc
python3.8 -m venv $venv
$venv/bin/pip install -q --upgrade pip==24.0
$venv/bin/pip install -q -r "$src/requirements-xmtvmc.txt"
$venv/bin/pip install -q --no-deps "$wheels"/caffe-1.0-py3-none-any.whl \
	"$wheels"/tvm_base-1.0-py3-none-any.whl "$wheels"/xmtvmc-1.0-py3-none-any.whl
sp=$($venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for p in "$src"/patches/*.patch; do patch -s -d "$sp" -p1 < "$p"; done

# Ubuntu names its serial HDF5 libhdf5_serial*; caffe was linked against the
# upstream names. xmnpu looks for its libraries in <venv>/lib/build, where the
# SDK's install.sh would have copied them.
extra=$venv/lib/extra && mkdir -p $extra
ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.103 $extra/libhdf5.so.103
ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.100 $extra/libhdf5_hl.so.100
mkdir -p $venv/lib/build && ln -sf "$sp"/xmnpu/*.so $venv/lib/build/
export LD_LIBRARY_PATH=$sp/xmnpu:$sp/caffe/lib:$extra
export PATH=$venv/bin:$PATH
