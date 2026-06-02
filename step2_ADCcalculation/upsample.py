#!/usr/bin/env python3
"""
Upsample a single NIfTI file from 480^3 to 960^3.
Typically used for FSL segmentation outputs (e.g. brain_pve masks).
"""
import os
import struct
import gzip
import numpy as np
from scipy.ndimage import zoom


def _read_nifti_raw(filepath):
    with gzip.open(filepath, 'rb') as f:
        header_data    = f.read(348)
        extension_bytes = f.read(4)
        data_bytes     = f.read()

    dim     = struct.unpack('<8h', header_data[40:56])
    pixdim  = struct.unpack('<8f', header_data[76:108])
    datatype = struct.unpack('<h', header_data[70:72])[0]
    nx, ny, nz = dim[1], dim[2], dim[3]

    dtype_map = {16: np.float32, 64: np.float64, 4: np.int16}
    dtype = dtype_map.get(datatype, np.float32)

    data = np.frombuffer(data_bytes, dtype=dtype)[:nx * ny * nz].reshape((nx, ny, nz))
    return header_data, extension_bytes, data, dim, pixdim


def _update_header_960(header_data, old_pixdim):
    header = bytearray(header_data)
    header[40:56] = struct.pack('<8h', 3, 960, 960, 960, 1, 1, 1, 1)
    header[76:108] = struct.pack('<8f',
        old_pixdim[0],
        old_pixdim[1] / 2.0, old_pixdim[2] / 2.0, old_pixdim[3] / 2.0,
        old_pixdim[4], old_pixdim[5], old_pixdim[6], old_pixdim[7])

    offset_shift = old_pixdim[1] / 4.0
    qform_code = struct.unpack('<h', header[252:254])[0]
    sform_code = struct.unpack('<h', header[254:256])[0]

    if qform_code > 0:
        qx, qy, qz = struct.unpack('<3f', header[268:280])
        header[268:280] = struct.pack('<3f',
            qx - offset_shift, qy - offset_shift, qz - offset_shift)

    if sform_code > 0:
        srow_x = struct.unpack('<4f', header[280:296])
        srow_y = struct.unpack('<4f', header[296:312])
        srow_z = struct.unpack('<4f', header[312:328])
        header[280:296] = struct.pack('<4f',
            srow_x[0]/2, srow_x[1]/2, srow_x[2]/2, srow_x[3] - offset_shift)
        header[296:312] = struct.pack('<4f',
            srow_y[0]/2, srow_y[1]/2, srow_y[2]/2, srow_y[3] - offset_shift)
        header[312:328] = struct.pack('<4f',
            srow_z[0]/2, srow_z[1]/2, srow_z[2]/2, srow_z[3] - offset_shift)

    return bytes(header)


def upsample_to_960(input_path, output_path, method='nearest'):
    """Upsample a 480^3 NIfTI file to 960^3.

    Args:
        input_path  : path to 480^3 .nii.gz file
        output_path : where to write the 960^3 output
        method      : 'nearest' for binary/label maps, 'linear' for probability maps
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[upsample_to_960] {os.path.basename(input_path)} -> {os.path.basename(output_path)}")

    header_data, ext_bytes, data, dim, pixdim = _read_nifti_raw(input_path)
    print(f"  Input  : {data.shape}  range [{data.min():.4f}, {data.max():.4f}]")

    order     = 0 if method == 'nearest' else 1
    upsampled = zoom(data, 2.0, order=order)
    print(f"  Output : {upsampled.shape}  method={method}")

    new_header = _update_header_960(header_data, pixdim)
    with gzip.open(output_path, 'wb', compresslevel=6) as f:
        f.write(new_header)
        f.write(ext_bytes)
        f.write(upsampled.astype(np.float32).tobytes())

    print("  Done.")