import math
import weakref

import torch
import numpy as np


def mod_crop(blob, N):
    if isinstance(blob, np.ndarray):
        # For numpy arrays, channels at the last dim
        h, w = blob.shape[-3:-1]
        nh = h - h % N
        nw = w - w % N
        return blob[..., :nh, :nw, :]
    else: 
        # For 4-D pytorch tensors, channels at the 2nd dim
        with torch.no_grad():
            h, w = blob.shape[-2:]
            nh = h - h % N
            nw = w - w % N
            return blob[..., :nh, :nw]


class HookHelper:
    def __init__(self, model, fetch_dict, out_dict, hook_type='forward_out'):
        self.model = weakref.proxy(model)
        self.fetch_dict = fetch_dict
        # Subclass the built-in list to make it weak referenceable
        class _list(list):
            pass
        for entry in self.fetch_dict.values():
            # entry is expected to be a string or a non-nested tuple
            if isinstance(entry, tuple):
                for key in entry:
                    out_dict[key] = _list()
            else:
                out_dict[entry] = _list()
        self.out_dict = weakref.WeakValueDictionary(out_dict)
        self._handles = []

        if hook_type not in ('forward_in', 'forward_out', 'backward_out'):
            raise NotImplementedError("Hook type is not implemented.")

        def _proto_hook(x, entry):
            # x should be a tensor or a tuple
            if isinstance(entry, tuple):
                for key, f in zip(entry, x):
                    self.out_dict[key].append(f.detach().clone())
            else:
                self.out_dict[entry].append(x.detach().clone())

        def _forward_in_hook(m, x, y, entry):
            # x is a tuple
            return _proto_hook(x[0] if len(x)==1 else x, entry)

        def _forward_out_hook(m, x, y, entry):
            # y is a tensor or a tuple
            return _proto_hook(y, entry)

        def _backward_out_hook(m, grad_in, grad_out, entry):
            # grad_out is a tuple
            return _proto_hook(grad_out[0] if len(grad_out)==1 else grad_out, entry)

        self._hook_func, self._reg_func_name = {
            'forward_in': (_forward_in_hook, 'register_forward_hook'),
            'forward_out': (_forward_out_hook, 'register_forward_hook'),
            'backward_out': (_backward_out_hook, 'register_backward_hook'),
        }[hook_type]

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name in self.fetch_dict:
                entry = self.fetch_dict[name]
                self._handles.append(
                    getattr(module, self._reg_func_name)(
                        lambda *args, entry=entry: self._hook_func(*args, entry=entry)
                    )
                )

    def __exit__(self, exc_type, exc_val, ext_tb):
        for handle in self._handles:
            handle.remove()