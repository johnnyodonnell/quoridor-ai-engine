"""Export a trained net to ONNX for browser inference via onnxruntime-web.

Output shape contract:
  input  "position"   float32 [1, 8, 9, 9]
  output "policy"     float32 [1, 140]   (raw logits, pre-softmax)
  output "value"      float32 [1]        (tanh-bounded, side-to-move POV)
"""

import argparse

import torch

from net import BOARD_SIZE, INPUT_CHANNELS, POLICY_SIZE, load_net


def export_module(net, onnx_path):
    """Export an in-memory QuoridorNet to ONNX."""
    was_training = net.training
    net.eval()
    # Match the net's device: the trainer keeps the net on CUDA.
    dummy = torch.zeros(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32,
                        device=next(net.parameters()).device)
    # dynamo=False forces the legacy tracing exporter, which inlines weights
    # in the ONNX file. The dynamo exporter externalizes them, which would
    # break browser inference (we want a single self-contained .onnx).
    torch.onnx.export(
        net,
        (dummy,),
        onnx_path,
        input_names=["position"],
        output_names=["policy", "value"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )
    if was_training:
        net.train()
    print(f"exported {onnx_path} (policy={POLICY_SIZE})", flush=True)


def export(ckpt_path, onnx_path):
    export_module(load_net(ckpt_path, "cpu"), onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.ckpt, args.out)
