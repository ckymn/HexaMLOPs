import os
import json
import requests
import subprocess
import time
import logging
import yaml
from yaml import FullLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTester:
    def __init__(self, config_file_path, model_config_file_path):
        self.config_data = self._read_yaml(config_file_path)
        self.model_config_data = self._read_yaml(model_config_file_path)
        self.models_to_test = self._get_models_to_test()
        
        if not self.config_data or 'docker' not in self.config_data or not self.config_data['docker']:
            raise ValueError("Docker registry information not found in config.yaml.")
        self.docker_registry = self.config_data['docker'][0]['registry']

    def _read_yaml(self, file_path):
        """Reads and returns data from a YAML file using PyYAML."""
        try:
            with open(file_path, 'r') as f:
                return yaml.load(f, Loader=FullLoader)
        except FileNotFoundError:
            logging.error(f"Error: The file {file_path} was not found.")
            raise
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing YAML file: {exc}")
            raise

    def _get_models_to_test(self):
        """Identifies and returns models marked for 'local' phase testing."""
        testable_models = []
        if not self.model_config_data:
            logging.warning("model.yaml appears to be empty or misconfigured.")
            return testable_models
            
        for model in self.model_config_data:
            for cluster in model.get('clusters', []):
                if cluster.get('phase') == 'local':
                    testable_models.append(model)
        return testable_models

    def run(self):
        if not self.models_to_test:
            logging.info("No models found for 'local' phase. Exiting.")
            return True

        for model_info in self.models_to_test:
            if not self._test_model(model_info):
                return False

        return True

    def _test_model(self, model_info):
        """Pulls a model's Docker image, runs a container, and performs an inference test."""
        model_name = model_info['name']
        version = model_info['environment']['version']
        image_name = f"{self.docker_registry}/{model_name}:{version}"
        container_id = None

        logging.info(f"Starting test for model: {model_name}, version: {version}")

        try:
            # Step 1: Pull Docker image
            logging.info(f"Pulling Docker image: {image_name}")
            subprocess.run(["docker", "pull", image_name], check=True)

            # Step 2: Run Docker container
            logging.info("Starting Docker container...")
            container_id = subprocess.check_output([
                "docker", "run", "-d", "-p", "8000:8000", "--rm", image_name
            ]).decode().strip()
            logging.info(f"Container started with ID: {container_id}")

            # Step 3: Wait for readiness
            api_url = f"http://localhost:8000/v2/health/ready"
            for i in range(30):
                try:
                    res = requests.get(api_url, timeout=5)
                    if res.status_code == 200:
                        logging.info("Container is ready.")
                        break
                except requests.exceptions.RequestException:
                    logging.info("Container not ready yet. Retrying...")
                time.sleep(2)
            else:
                logging.error("Container failed to become ready.")
                return False
            
            # Step 4: Run inference test using dynamic test data path
            logging.info("Running inference test...")
            inference_url = f"http://localhost:8000/v2/models/{model_name}/versions/{version}/infer"
            
            test_data_path = model_info.get('test_data_path')
            if not test_data_path:
                logging.error(f"'test_data_path' is not specified for model '{model_name}' in model.yaml.")
                return False

            if not os.path.exists(test_data_path):
                logging.error(f"Test data file not found at the specified path: {test_data_path}")
                return False

            with open(test_data_path, 'r') as f:
                payload = json.load(f)

            res = requests.post(inference_url, json=payload)
            if res.status_code != 200:
                logging.error(f"Inference request failed with status code {res.status_code}: {res.text}")
                return False
            
            # Step 5: Validate the output
            logging.info("Validating model output...")
            output = res.json()
            if not self._validate_output(output):
                logging.error("Output validation failed.")
                return False
            
            logging.info(f"Model test for '{model_name}' passed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Docker command failed: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return False
        finally:
            if container_id:
                logging.info(f"Stopping container: {container_id}")
                subprocess.run(["docker", "stop", container_id], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
    def _validate_output(self, output):
        # Validate output shape
        if not 'outputs' in output or len(output['outputs']) != 1:
            logging.error("Invalid output format: 'outputs' not found or wrong length.")
            return False
        
        output_data = output['outputs'][0]
        if output_data['shape'] != [200, 1]:
            logging.error(f"Incorrect output shape: {output_data['shape']}. Expected: [200, 1].")
            return False

        # Validate data range based on your test file
        if not 'data' in output_data or len(output_data['data']) != 200:
            logging.error("Invalid output data: 'data' not found or wrong length.")
            return False

        for value in output_data['data']:
            if not (-500 <= value <= 500):
                logging.error(f"Value {value} is outside the expected range [-500, 500].")
                return False
        
        return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Model testing script for HexaMLOPs.")
    parser.add_argument('--config-file-path', required=True, help="Path to the main config.yaml file.")
    parser.add_argument('--model-config-file-path', required=True, help="Path to the model.yaml file.")
    args = parser.parse_args()

    try:
        tester = ModelTester(
            config_file_path=args.config_file_path,
            model_config_file_path=args.model_config_file_path
        )
        if not tester.run():
            exit(1)
    except Exception as e:
        logging.error(f"Failed to run ModelTester: {e}")
        exit(1)