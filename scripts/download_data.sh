#!/bin/bash
# Script to download datasets for ST-Trace

# Usage: ./scripts/download_data.sh <dataset-name>
# dataset-name: nlpr, dukemtmc, cityflow, all

DATASET=$1
TARGET_DIR="./data/raw"

mkdir -p $TARGET_DIR

download_nlpr() {
    echo "Downloading NLPR_MCT..."
    # Note: NLPR_MCT requires request from authors
    echo "NLPR_MCT must be obtained from the authors of 'ECCV 2014: Multi-Camera Tracking..."
    echo "Please download manually and place it in $TARGET_DIR/NLPR_MCT/"
}

download_dukemtmc() {
    echo "Downloading DukeMTMC-videoReID..."
    cd $TARGET_DIR
    if command -v wget >/dev/null 2>&1; then
        wget https://cvg.citg.tudelft.nl/Data/DukeMTMC/DukeMTMC-vid4reid.zip
    elif command -v curl >/dev/null 2>&1; then
        curl -O https://cvg.citg.tudelft.nl/Data/DukeMTMC/DukeMTMC-vid4reid.zip
    else
        echo "Neither wget nor curl found. Please download manually from:"
        echo "https://cvg.citg.tudelft.nl/Data/DukeMTMC/DukeMTMC-vid4reid.zip"
        echo "and place it in $TARGET_DIR/DukeMTMC-vid4reid.zip"
        exit 1
    fi
    unzip DukeMTMC-vid4reid.zip
    rm DukeMTMC-vid4reid.zip
    echo "Done."
}

download_cityflow() {
    echo "Downloading CityFlow..."
    # CityFlow can be downloaded from:
    # https://github.com/facebookresearch/CityFlow
    echo "Please follow instructions at https://github.com/facebookresearch/CityFlow"
    echo "and place extracted data in $TARGET_DIR/CityFlow/"
}

case $DATASET in
nlpr)
        download_nlpr
        ;;
    dukemtmc)
        download_dukemtmc
        ;;
    cityflow)
        download_cityflow
        ;;
    all)
        download_nlpr
        download_dukemtmc
        download_cityflow
        ;;
    *)
        echo "Usage: ./scripts/download_data.sh [nlpr|dukemtmc|cityflow|all]"
        exit 1
        ;;
esac
