"""Exact four-beam NF4 products for the locally validated Ada/BF16 backend.

The native kernel changes storage layout and launch scheduling, not quantization.
Every projection is checked against bitsandbytes before its forward is replaced.
"""
import ctypes
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


def load_library():
    if os.name != 'nt' or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError('The native decoder is validated only on Windows Ada GPUs')
    source = Path(__file__).parent / 'kernels' / 'nf4_decode.cu'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(__file__).resolve().parent.parent / 'artifacts' / 'native'
    directory.mkdir(parents=True, exist_ok=True)
    library_path = directory / f'nf4_decode_sm89_{digest}.dll'
    if not library_path.exists():
        program_files = Path(os.environ.get('ProgramFiles', r'C:\Program Files'))
        nvcc = shutil.which('nvcc')
        if not nvcc:
            candidates = sorted(program_files.glob('NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe'))
            nvcc = str(candidates[-1]) if candidates else None
        cl = shutil.which('cl')
        if not cl:
            candidates = sorted(program_files.glob('Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
            cl = str(candidates[-1]) if candidates else None
        if not nvcc or not cl:
            raise RuntimeError('CUDA nvcc and Visual Studio C++ tools are required for native decode')
        logger.info('Building the local NF4 decoder kernel')
        result = subprocess.run(
            [nvcc, '-shared', '-O3', '-arch=sm_89', '-ccbin', str(Path(cl).parent),
             '-o', str(library_path), str(source)],
            cwd=directory, capture_output=True, text=True, timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode:
            raise RuntimeError(f'Native decoder build failed: {result.stderr[-2000:]}')
    library = ctypes.CDLL(str(library_path))
    library.pack_nf4.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    library.unpack_nf4.argtypes = library.pack_nf4.argtypes
    library.launch_register.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p] + [ctypes.c_int] * 2
    return library


class OriginalLayout:
    """Reconstruct the exact original bytes only while the stock forward needs them."""
    def __init__(self, library, module, packed):
        self.library, self.module, self.packed = library, module, packed
        self.original = module.forward
        self.shape = module.weight.shape
        if not torch.equal(module.weight, self.unpack()):
            raise RuntimeError('NF4 layout permutation did not preserve every weight byte')

    def unpack(self):
        rows, cols = self.module.weight.quant_state.shape
        weight = torch.empty(self.shape, device=self.packed.device, dtype=torch.uint8)
        rc = self.library.unpack_nf4(self.packed.data_ptr(), weight.data_ptr(), rows, cols,
                                    torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'NF4 unpacking failed: CUDA error {rc}')
        return weight

    def release_original(self):
        self.module.weight.data = self.packed.view(torch.uint8).view(self.shape)

    def restore_original(self):
        self.module.weight.data = self.unpack()

    def forward(self, x):
        original_layout = self.unpack()
        try:
            self.module.weight.data = original_layout
            return self.original(x)
        finally:
            self.release_original()


class PackedProduct:
    def __init__(self, library, modules):
        import bitsandbytes as bnb
        self.library = library
        self.in_features = modules[0].in_features
        self.sizes = [module.out_features for module in modules]
        self.originals = [module.forward for module in modules]
        packed, scales = [], []
        for module in modules:
            state = module.weight.quant_state
            if (module.in_features != self.in_features or state.quant_type != 'nf4'
                    or state.blocksize != 64 or not state.nested or module.bias is not None
                    or module.compute_dtype != torch.bfloat16 or state.dtype != torch.bfloat16):
                raise RuntimeError('Unsupported projection configuration')
            rows, cols = state.shape
            if rows % 128 or cols % 256:
                raise RuntimeError('Unsupported projection dimensions')
            weight = torch.empty(rows * cols // 8, device=module.weight.device, dtype=torch.int32)
            rc = library.pack_nf4(module.weight.data_ptr(), weight.data_ptr(), rows, cols,
                                  torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(f'NF4 packing failed: CUDA error {rc}')
            scale = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
            scale = scale.reshape(rows // 16, 16, cols // 64).transpose(1, 2).contiguous().flatten()
            packed.append(weight)
            scales.append(scale)
        self.weight = torch.cat(packed) if len(packed) > 1 else packed[0]
        self.scales = torch.cat(scales) if len(scales) > 1 else scales[0]
        self.state = SimpleNamespace(shape=(sum(self.sizes), self.in_features), code=modules[0].weight.quant_state.code)
        if any(not torch.equal(module.weight.quant_state.code, self.state.code) for module in modules):
            raise RuntimeError('Projection codebooks differ')
        self.warps = 4 if sum(self.sizes) >= 10240 else 1
        self.outputs = None
        self.input_pointer = None
        for amplitude in (.1, 1, 10):
            sample = torch.randn((4, 1, self.in_features), dtype=torch.bfloat16, device=self.weight.device) * amplitude
            expected = [forward(sample) for forward in self.originals]
            actual = self.product(sample)
            if not all(torch.equal(a, b) for a, b in zip(expected, actual)):
                raise RuntimeError('Native NF4 product failed exact numerical validation')

    def product(self, x):
        rows, cols = self.state.shape
        output = torch.empty((4, 1, rows), dtype=x.dtype, device=x.device)
        rc = self.library.launch_register(
            x.data_ptr(), self.weight.data_ptr(), self.scales.data_ptr(), self.state.code.data_ptr(),
            output.data_ptr(), rows, cols, torch.cuda.current_stream().cuda_stream, self.warps, 16,
        )
        if rc:
            raise RuntimeError(f'NF4 product failed: CUDA error {rc}')
        return tuple(part.contiguous() for part in output.split(self.sizes, dim=-1))

    def forward(self, index):
        def call(x):
            if (x.shape != (4, 1, self.in_features) or x.dtype != torch.bfloat16
                    or not x.is_contiguous() or torch.is_grad_enabled()):
                return self.originals[index](x)
            if index == 0:
                self.outputs = self.product(x)
                self.input_pointer = x.data_ptr()
            if self.input_pointer != x.data_ptr():
                return self.originals[index](x)
            return self.outputs[index]
        return call


def install_products(model, patches):
    library = load_library()
    products = []
    for layer in model.model.layers:
        attention = layer.self_attn
        groups = [[attention.q_proj, attention.k_proj, attention.v_proj],
                  [attention.o_proj], [layer.mlp.gate_proj], [layer.mlp.up_proj], [layer.mlp.down_proj]]
        for modules in groups:
            product = PackedProduct(library, modules)
            offset = 0
            for index, module in enumerate(modules):
                if os.getenv('KOSHUR_SINGLE_WEIGHT_LAYOUT', '1') != '0':
                    size = module.weight.numel() // 4
                    layout = OriginalLayout(library, module, product.weight[offset:offset+size])
                    offset += size
                    patches.append(layout.restore_original)
                    layout.release_original()
                    product.originals[index] = layout.forward
                patches.append((module, 'forward', module.forward))
                module.forward = product.forward(index)
            products.append(product)
    return products
