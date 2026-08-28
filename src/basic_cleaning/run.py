#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the results to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
  
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = args.input_artifact
    local_path = wandb.use_artifact(artifact_local_path).file()
    df = pd.read_csv(local_path)

    # Apply basic data cleaning
    min_price = args.min_price
    max_price = args.max_price

    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Cleaned data has %d rows and %d columns", *df.shape)

    # Save the cleaned dataframe to a new CSV file
    df.to_csv("clean_sample.csv", index=False)

    # Create and log the new artifact and upload to wandb
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name and version of the raw input W&B artifact, for example 'sample.csv:latest'.",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output W&B artifact.",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="W&B artifact type for the cleaned dataset, for example 'clean_data'.",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the cleaned output artifact.",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed listing price. Rows below this value are removed.",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed listing price. Rows above this value are removed.",
        required=True,
    )



    args = parser.parse_args()

    go(args)
