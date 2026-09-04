"""
Roboflow Dataset Downloader Script for License Plate Detection (Model 2)
"""

import sys

def download_dataset():
    try:
        from roboflow import Roboflow
        print("Initializing Roboflow download...")
        rf = Roboflow(api_key="8wWvbViDNbyDWKgFMV8k")
        
        # Try primary project / version
        try:
            project = rf.workspace("samrat-sahoo").project("license-plates-us-eu")
            version = project.version(1)
            dataset = version.download("yolov8")
            print(f"Dataset successfully downloaded to: {dataset.location}")
            return dataset.location
        except Exception as e:
            print(f"Roboflow primary project download notice: {e}")
            print("Attempting fallback Roboflow public LP dataset...")
            project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
            version = project.version(1)
            dataset = version.download("yolov8")
            print(f"Fallback dataset successfully downloaded to: {dataset.location}")
            return dataset.location

    except Exception as e:
        print(f"Error during dataset download: {e}")
        return None

if __name__ == "__main__":
    download_dataset()
