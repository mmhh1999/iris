"""Native renderer failures must reach Python before invalid buffers are traced."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <cstring>
#include <iostream>
using cudaError_t = int;
using OptixResult = int;
constexpr int cudaSuccess = 0;
constexpr int OPTIX_SUCCESS = 0;
const char* cudaGetErrorString(int) { return "allocation failed"; }
#include "HEADER"

int main() {
    int failures = 0;
    bool continued = false;
    try { CUDA_CHECK(2); continued = true; }
    catch (const std::runtime_error& e) {
        if (std::strstr(e.what(), "allocation failed")) ++failures;
    }
    try { OPTIX_CHECK(7001); continued = true; }
    catch (const std::runtime_error& e) {
        if (std::strstr(e.what(), "Optix call")) ++failures;
    }
    char log[64] = "outputBuffer is 0";
    size_t sizeof_log = sizeof(log);
    try { OPTIX_CHECK_LOG(7001); continued = true; }
    catch (const std::runtime_error& e) {
        if (std::strstr(e.what(), "outputBuffer is 0")) ++failures;
    }
    CUDA_CHECK(cudaSuccess);
    OPTIX_CHECK(OPTIX_SUCCESS);
    OPTIX_CHECK_LOG(OPTIX_SUCCESS);
    CUDA_CHECK_NO_THROW(2);
    OPTIX_CHECK_NO_THROW(7001);
    if (continued || failures != 3) return 1;
    std::cout << "All native errors propagated; success calls accepted\n";
}
'''


class NativeErrorsTest(unittest.TestCase):
    def test_cuda_and_optix_failures_abort_before_tracing(self):
        compiler = shutil.which('g++')
        if compiler is None:
            candidate = ROOT / '.toolchain/compiler/bin/x86_64-conda-linux-gnu-g++'
            compiler = str(candidate) if candidate.exists() else None
        if compiler is None:
            self.skipTest('C++ compiler is unavailable')
        headers = [
            'threedgrt_tracer/include/3dgrt/cuoptixMacros.h',
            'threedgptir_tracer/include/3dgptir/cuoptixMacros.h',
        ]
        for header in headers:
            with self.subTest(header=header), tempfile.TemporaryDirectory() as d:
                source = pathlib.Path(d) / 'native_errors.cpp'
                binary = pathlib.Path(d) / 'native_errors'
                path = ROOT / 'third_party/ptir_gs' / header
                source.write_text(HARNESS.replace('HEADER', str(path)))
                subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)],
                               check=True, capture_output=True, text=True)
                result = subprocess.run([str(binary)], check=True,
                                        capture_output=True, text=True)
                self.assertIn('All native errors propagated', result.stdout)


if __name__ == '__main__':
    unittest.main()
