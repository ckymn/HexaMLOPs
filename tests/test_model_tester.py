import unittest
from unittest.mock import patch, MagicMock
import json
import subprocess
import os
import requests
import time

# Importing the HexaMLOPs's own module
from hexa_mlops.model_tester.model_tester import ModelTester

class TestModelTester(unittest.TestCase):
    
    # Mock method to simulate actual file reading operations
    @patch('hexa_mlops.model_tester.model_tester.ModelTester._read_yaml')
    def setUp(self, mock_read_yaml):
        # Defining fake config.yaml and model.yaml data
        self.mock_config_data = {
            'docker': [{'registry': 'mock-registry', 'name': 'dev'}],
            'environments': [{'name': 'custom-24.12-onnx-py', 'version': 7}]
        }
        self.mock_model_config_data = [
            {
                'name': 'mlpricer-ua-onnx-noblob',
                'environment': {'name': 'custom-24.12-onnx-py', 'version': 1},
                'clusters': [{'phase': 'local'}]
            }
        ]
        
        # Making the _read_yaml method return the fake data when called
        mock_read_yaml.side_effect = [self.mock_config_data, self.mock_model_config_data]
        
        # Initializing the ModelTester class
        self.tester = ModelTester(config_file_path='mock_config.yaml', model_config_file_path='mock_model.yaml')

    # Testing a successful scenario by mocking Docker commands and HTTP requests
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    @patch('requests.get')
    @patch('requests.post')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    def test_run_success(self, mock_json_load, mock_open, mock_exists, mock_post, mock_get, mock_run, mock_check_output):
        
        # Simulating that Docker ran successfully
        mock_check_output.return_value = b'container_id_123'
        
        # Simulating a successful health check
        mock_get.return_value.status_code = 200
        
        # Simulating successful output from the model
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'outputs': [{'shape': [200, 1], 'data': [100.0] * 200}]
        }
        
        # Simulating the reading of test data
        mock_json_load.return_value = {'inputs': []}
        
        # Running the run method and expecting it to return True
        result = self.tester.run()
        self.assertTrue(result)
        
        # Checking that the Docker commands were called with the correct arguments
        mock_check_output.assert_called_with(['docker', 'run', '-d', '-p', '8000:8000', '--rm', 'mock-registry/mlpricer-ua-onnx-noblob:1'])
        mock_post.assert_called_once()
        mock_run.assert_any_call(['docker', 'stop', 'container_id_123'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # Testing the scenario where the Docker container fails to start
    @patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'docker run'))
    @patch('subprocess.run')
    def test_run_docker_fail(self, mock_run, mock_check_output):
        result = self.tester.run()
        self.assertFalse(result)
        
    # Testing the scenario where the model does not provide the correct output
    @patch('subprocess.check_output', return_value=b'container_id_123')
    @patch('subprocess.run')
    @patch('requests.get', return_value=MagicMock(status_code=200))
    @patch('requests.post', return_value=MagicMock(status_code=200, json=lambda: {'outputs': [{'shape': [100, 1]}]}))
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load', return_value={'inputs': []})
    def test_run_validation_fail(self, mock_json_load, mock_open, mock_exists, mock_post, mock_get, mock_run, mock_check_output):
        result = self.tester.run()
        self.assertFalse(result)
        mock_run.assert_any_call(['docker', 'stop', 'container_id_123'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    unittest.main()