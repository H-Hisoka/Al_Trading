
# File: models/model_registry.py

import os
import json
import shutil
import hashlib
import joblib

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

class RegistryConfig:

    def __init__(self):

        self.store_dir = Path("models_store")

        self.versions_dir = (
            self.store_dir / "versions"
        )

        self.registry_path = (
            self.store_dir / "registry.json"
        )

        self.active_model_path = (
            self.store_dir / "swing_model.joblib"
        )


# ============================================================
# 2. REGISTRY MANAGER
# ============================================================

class ModelRegistry:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else RegistryConfig()
        )

        self._create_directories()

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    def _create_directories(self):

        self.config.store_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.config.versions_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # LOAD REGISTRY
    # --------------------------------------------------------

    def _load_registry(self):

        path = self.config.registry_path

        if not path.exists():

            return {
                "active_version": None,
                "versions": []
            }

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            registry = json.load(file)

        if not isinstance(registry, dict):
            raise ValueError(
                "registry.json không hợp lệ"
            )

        registry.setdefault(
            "active_version",
            None
        )

        registry.setdefault(
            "versions",
            []
        )

        return registry

    # --------------------------------------------------------
    # SAVE REGISTRY ATOMICALLY
    # --------------------------------------------------------

    def _save_registry(self, registry):

        path = self.config.registry_path

        temp_path = path.with_suffix(
            ".json.tmp"
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                registry,
                file,
                ensure_ascii=False,
                indent=4
            )

        os.replace(
            temp_path,
            path
        )

    # --------------------------------------------------------
    # GENERATE VERSION
    # --------------------------------------------------------

    def _generate_version(self, registry):

        existing_versions = [
            item["version"]
            for item in registry["versions"]
        ]

        number = 1

        while True:

            version = f"v{number:04d}"

            if version not in existing_versions:
                return version

            number += 1

    # --------------------------------------------------------
    # FILE HASH
    # --------------------------------------------------------

    @staticmethod
    def _calculate_hash(file_path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        model_path=None,
        metrics=None,
        description=""
    ):

        if model_path is None:

            model_path = (
                self.config.active_model_path
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}"
            )

        artifact = joblib.load(model_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Model artifact không hợp lệ"
            )

        if "model" not in artifact:
            raise ValueError(
                "Artifact thiếu model"
            )

        if "metadata" not in artifact:
            raise ValueError(
                "Artifact thiếu metadata"
            )

        registry = self._load_registry()

        version = self._generate_version(
            registry
        )

        version_dir = (
            self.config.versions_dir / version
        )

        version_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        version_model_path = (
            version_dir / "model.joblib"
        )

        shutil.copy2(
            model_path,
            version_model_path
        )

        model_hash = self._calculate_hash(
            version_model_path
        )

        metadata = artifact["metadata"]

        calibration_metadata = artifact.get(
            "calibration_metadata"
        )

        record = {

            "version": version,

            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "model_path": str(
                version_model_path
            ),

            "sha256": model_hash,

            "description": description,

            "classes": metadata.get(
                "classes",
                []
            ),

            "feature_columns": metadata.get(
                "feature_columns",
                []
            ),

            "target": metadata.get(
                "target"
            ),

            "horizon_bars": metadata.get(
                "horizon_bars"
            ),

            "best_iteration": metadata.get(
                "best_iteration"
            ),

            "calibrated": (
                calibration_metadata is not None
            ),

            "calibration_method": (
                calibration_metadata.get("method")
                if calibration_metadata
                else None
            ),

            "metrics": metrics or {},

        }

        registry["versions"].append(
            record
        )

        self._save_registry(
            registry
        )

        print(
            f"Đã đăng ký model: {version}"
        )

        return record

    # --------------------------------------------------------
    # LIST MODELS
    # --------------------------------------------------------

    def list_models(self):

        registry = self._load_registry()

        return registry["versions"]

    # --------------------------------------------------------
    # GET MODEL INFORMATION
    # --------------------------------------------------------

    def get_model_info(self, version):

        registry = self._load_registry()

        for record in registry["versions"]:

            if record["version"] == version:
                return record

        raise ValueError(
            f"Không tìm thấy model {version}"
        )

    # --------------------------------------------------------
    # ACTIVATE MODEL
    # --------------------------------------------------------

    def activate_model(self, version):

        registry = self._load_registry()

        record = self.get_model_info(
            version
        )

        source_path = Path(
            record["model_path"]
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model: {source_path}"
            )

        temp_path = (
            self.config.active_model_path.with_suffix(
                ".joblib.tmp"
            )
        )

        shutil.copy2(
            source_path,
            temp_path
        )

        os.replace(
            temp_path,
            self.config.active_model_path
        )

        registry["active_version"] = version

        self._save_registry(
            registry
        )

        print(
            f"Đã kích hoạt model: {version}"
        )

        return record

    # --------------------------------------------------------
    # GET ACTIVE MODEL
    # --------------------------------------------------------

    def get_active_model(self):

        registry = self._load_registry()

        version = registry.get(
            "active_version"
        )

        if version is None:
            raise RuntimeError(
                "Chưa có model nào được kích hoạt"
            )

        return self.get_model_info(
            version
        )

    # --------------------------------------------------------
    # LOAD MODEL BY VERSION
    # --------------------------------------------------------

    def load_model(self, version=None):

        if version is None:

            record = self.get_active_model()

        else:

            record = self.get_model_info(
                version
            )

        model_path = Path(
            record["model_path"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}"
            )

        return joblib.load(
            model_path
        )

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    def compare_models(self):

        records = self.list_models()

        if not records:
            return []

        results = []

        for record in records:

            results.append({

                "version": record["version"],

                "created_at": record["created_at"],

                "calibrated": record["calibrated"],

                "calibration_method": (
                    record["calibration_method"]
                ),

                "metrics": record["metrics"],

            })

        return results


# ============================================================
# 3. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    registry = ModelRegistry()

    models = registry.list_models()

    if not models:

        print("Registry chưa có model nào.")

    else:

        print("\n===== MODEL REGISTRY =====")

        for model in models:

            print(
                f"\nVersion: {model['version']}"
            )

            print(
                f"Created: {model['created_at']}"
            )

            print(
                f"Calibrated: {model['calibrated']}"
            )

            print(
                f"Metrics: {model['metrics']}"
            )

        print(
            "\nActive model:",
            registry._load_registry()["active_version"]
        )
