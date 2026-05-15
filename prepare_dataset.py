# Purpose: Implements prepare_dataset.py in the PoisonedRAG project.

from beir import util
import os
import argparse
import glob


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--legacy_beir", action="store_true", help="Download legacy BEIR datasets (nq/msmarco/hotpotqa)")
    args = parser.parse_args()

    if not args.legacy_beir:
        print("Current project defaults to MedRAG corpora: pubmed/statpearls/textbooks.")
        print("Please place corpus files under datasets/<corpus>/chunk and index directories, or set MEDRAG_DATA_DIR.")
        print("Use --legacy_beir if you still need nq/msmarco/hotpotqa downloads.")
        return

    datasets = ['nq', 'msmarco', 'hotpotqa']
    for dataset in datasets:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = os.path.join(out_dir, dataset)
        if not os.path.exists(data_path):
            util.download_and_unzip(url, out_dir)

    for zip_path in glob.glob(os.path.join("datasets", "*.zip")):
        try:
            os.remove(zip_path)
        except OSError:
            pass


if __name__ == '__main__':
    main()