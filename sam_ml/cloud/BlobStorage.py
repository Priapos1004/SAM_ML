import os

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient


class BlobStorage:
    def __init__(self, azure_blob_connection_string: str, azure_blob_data_container: str, local_interim_step_path: str = "BlobStorage_interim_step.csv"):
        self.blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)
        self.container = azure_blob_data_container
        self.local_interim_step_path = local_interim_step_path

    def upload_csv(self, local_path: str, target_path: str):
        blob_client = self.blob_service_client.get_blob_client(container=self.container, blob=target_path)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data=data)

    def download_csv(self, target_path: str, local_path: str):
        blob_client = self.blob_service_client.get_blob_client(container=self.container, blob=target_path)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

    def delete_blob(self, target_path: str):
        blob_client = self.blob_service_client.get_blob_client(container=self.container, blob=target_path)
        blob_client.delete_blob(delete_snapshots = "include")

    def upload_df(self, df: pd.DataFrame, target_path: str):
        df.to_csv(self.local_interim_step_path)
        self.upload_csv(self.local_interim_step_path, target_path)
        os.remove(self.local_interim_step_path)

    def download_df(self, target_path: str) -> pd.DataFrame:
        self.download_csv(target_path, self.local_interim_step_path)
        df = pd.read_csv(self.local_interim_step_path, index_col=0)
        os.remove(self.local_interim_step_path)
        return df

    def upload_np_ndarray(self, array: np.ndarray, target_path: str):
        np.savetxt(self.local_interim_step_path, array, delimiter=",")
        self.upload_csv(self.local_interim_step_path, target_path)
        os.remove(self.local_interim_step_path)

    def download_np_ndarray(self, target_path: str) -> np.ndarray:
        self.download_csv(target_path, self.local_interim_step_path)
        array = np.genfromtxt(self.local_interim_step_path, delimiter=',')
        os.remove(self.local_interim_step_path)
        return array
