# ST-Trace: Neural Graph Search with Spatio-Temporal Priors for Efficient Multi-Camera Tracking

[//]: # "Official implementation of ST-Trace for CCF-B+ conference submission."

## Overview

ST-Trace proposes **ST-ANBS (Adaptive Neural Beam Search)**, a neural graph search approach that leverages learned spatio-temporal priors to achieve efficient multi-camera tracking in large-scale surveillance networks.

Key contributions:

- **Theoretical guarantee**: Probability bound on suboptimality with linear complexity \(O(KBD_{\text{max}})\) vs exponential exhaustive search
- **Spatio-temporal contrastive learning**: Novel ReID training paradigm that uses camera topology as supervision
- **Coarse-to-fine retrieval**: Cascaded pipeline that reduces processed frames by ~87% while maintaining accuracy
- **5.6× - 6× speedup** over state-of-the-art GNN-based trackers

## Installation

```bash
git clone https://github.com/Listcapture/ST-trace.git
cd ST-trace
pip install -r requirements.txt
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Data Preparation

Download the three benchmark datasets:
- NLPR_MCT
- DukeMTMC-videoReID
- CityFlow

See [docs/data_preparation.md](docs/data_preparation.md) for detailed instructions.

## Quick Start

```bash
# Preprocess datasets
python scripts/preprocess.py --config configs/nlpr_mct.yaml

# Train transition model for ST-ANBS
python scripts/train_transition.py --config configs/nlpr_mct.yaml

# Train ReID with spatio-temporal contrastive learning
python scripts/train_reid.py --config configs/nlpr_mct.yaml

# Evaluate full pipeline
python scripts/evaluate.py --config configs/nlpr_mct.yaml
```

See [docs/quickstart.md](docs/quickstart.md) for more examples.

## Project Structure

```
st_trace/
├── data/         # Dataset loading and camera graph
├── models/       # Model components (transition net, ReID, detector)
├── search/       # Graph search algorithms (ST-ANBS, baselines)
├── tracking/     # Full tracking pipeline
├── evaluation/   # Evaluation metrics
├── visualization/# Visualization tools
└── utils/        # Utilities
```

## Experimental Results

Coming soon...

## Citation

```
@inproceedings{sttrace202x,
  title={Neural Graph Search with Spatio-Temporal Priors for Efficient Multi-Camera Tracking},
  author={Your Name},
  booktitle={Conference Name},
  year={202x}
}
```

## License

MIT License
