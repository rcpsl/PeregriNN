import time
import os
import sys
from dnnv.dnnv.nn import parse
from dnnv.ddnv.nn.transformers.simplifiers import (simplify, ReluifyMaxPool)
from pathlib import Path

def main():
    """main entry point"""

    assert len(sys.argv) == 3, "expected 2 arguments: [Category][input .onnx filename]"

    category = sys.argv[1]
    if (category not in ['cifar_biasfield']):
        return

    onnx_filename = sys.argv[2]
    op_graph = parse(Path(onnx_filename))

    print("Simplifying network...")
    t = time.perf_counter()
    try:
        simplified_op_graph1 = simplify(op_graph, ReluifyMaxPool(op_graph))
    except Exception as e:
        print("Couldn't simplify network:",str(e))
    diff = time.perf_counter() - t
    print(f"Simplify runtime: {diff}")
    
    print("Exporting...")
    t = time.perf_counter()
    dirname = 'vnn_simplified_networks'
    if(not os.path.isdir(dirname)):
        os.mkdir(dirname)
    fname = onnx_filename.split('/')[-1]
    fname = fname.split('.')[0]
    fpath = os.path.join(dirname, fname) + '_simplified.onnx'
    print(f'Exporting simplified network to {fpath}')
    simplified_op_graph1.export_onnx(fpath)
    diff = time.perf_counter() - t
    print(f"Export runtime: {diff}")

if __name__ == "__main__":
    main()
