#!/usr/bin/env python
"""
Preprocess dataset annotations for ST-Trace.

Usage:
    python scripts/preprocess.py --config configs/nlpr_mct.yaml
"""
import argparse
import yaml
import os
from st_trace.data.nlpr_mct import NLPRMCTDataset
from st_trace.data.dukemtmc import DukeMTMCDataset
from st_trace.data.cityflow import CityFlowDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--root', type=str, default='./data/raw', help='Dataset root')
    parser.add_argument('--output', type=str, default='./data/processed', help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config.get('dataset', 'nlpr_mct')
    root_path = os.path.join(args.root, dataset_name)
    output_path = os.path.join(args.output, f'{dataset_name}_{config["split"]}.json')

    # Create dataset (this loads and processes annotations)
    if dataset_name == 'nlpr_mct':
        dataset = NLPRMCTDataset(root_path=root_path, split=config['split'])
    elif dataset_name == 'dukemtmc':
        dataset = DukeMTMCDataset(root_path=root_path, split=config['split'])
    elif dataset_name == 'cityflow':
        dataset = CityFlowDataset(root_path=root_path, split=config['split'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Save processed annotations
    os.makedirs(args.output, exist_ok=True)
    dataset.save_annotations(output_path)

    print(f"Processed {len(dataset.trajectories)} trajectories")
    print(f"Saved to {output_path}")
    print(f"Camera graph has {dataset.camera_graph.num_cameras} cameras, {dataset.camera_graph.num_edges} edges")


if __name__ == '__main__':
    main()
