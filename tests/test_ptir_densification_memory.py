"""Exact tensor-copy equivalence for the lower-memory split implementation."""
import ast
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_helper():
    tree = ast.parse((ROOT / 'third_party/ptir_gs/threedgrut/strategy/base.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'indexed_append')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'indexed_append', 'exec'), scope)
    return scope['indexed_append']


class DensificationMemoryTest(unittest.TestCase):
    def test_split_and_adam_moments_are_bitwise_identical(self):
        append = load_helper()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for columns in [1, 3, 4, 45]:
                for kept in [[], [0], [0, 2, 4, 9], list(range(10))]:
                    for dtype in [torch.float32, torch.float64]:
                        with self.subTest(device=device, columns=columns, kept=kept, dtype=dtype):
                            source = torch.arange(10 * columns, device=device, dtype=dtype).reshape(10, columns)
                            indices = torch.tensor(kept, device=device, dtype=torch.long)
                            tail = source[[1, 3]].repeat(2, 1)
                            for new_rows in [tail, torch.zeros_like(tail), tail[:0]]:
                                expected = torch.cat([source[indices], new_rows])
                                actual = append(source, indices, new_rows)
                                self.assertTrue(torch.equal(expected, actual))
                                self.assertNotEqual(actual.data_ptr(), source.data_ptr())


if __name__ == '__main__':
    unittest.main()
