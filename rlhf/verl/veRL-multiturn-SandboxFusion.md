# VeRL-multiturn-SandboxFusion

# Get Started

---

Sandbox Fusion is a versatile platform for **code execution** and **evaluation** that supports over 20 programming languages and more than 10 coding-related evaluation datasets. Built for cloud deployment, it offers two main functions: running code and evaluating solution correctness. The platform features both *script* and *Jupyter* execution modes, with customizable security isolation levels set through a *YAML file*. For each execution, it creates a temporary directory that's automatically deleted afterward, and handles file transfers using base64 encoding.

The code sandbox service primarily offers two functions:

- **running code**
- **evaluating the correctness of problems**

Supported programming languages:

![Screenshot 2025-03-25 at 7.04.54 PM.png](./img/sandbox_supported_languages.png)

Implemented open-source datasets:

![Screenshot 2025-03-25 at 7.05.28 PM.png](./img/sandbox_supported_datasets.png)

# Local Deployment

---

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20241204
```

# Usage

---

## Code Sandbox

<aside>

> 💡 Tip:
> You can find a simple playground page at [http://localhost:8080/? SandboxFusion/playground/sandbox](http://localhost:8080/SandboxFusion/playground/sandbox)

</aside>

Execute the following command in the shell to request the sandbox to run a Python code snippet:

```bash
curl 'http://localhost:8080/run_code' \
  -H 'Content-Type: application/json' \
  --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
```

Sample output:

```json
{
  "status": "Success",
  "message": "",
  "compile_result": null,
  "run_result": {
    "status": "Finished",
    "execution_time": 0.016735315322875977,
    "return_code": 0,
    "stdout": "Hello, world!\\n",
    "stderr": ""
  },
  "executor_pod_name": null,
  "files": {}
}
```

You can also make a similar request using a `Python script`. Here's an example of running C++ code:

```python
import requests
import json

response = requests.post('http://localhost:8080/run_code', json={
    'code': '''
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
''',
    'language': 'cpp',
})

print(json.dumps(response.json(), indent=2))
```

Sample output:

```json
{
  "status": "Success",
  "message": "",
  "compile_result": {
    "status": "Finished",
    "execution_time": 0.45870447158813477,
    "return_code": 0,
    "stdout": "",
    "stderr": ""
  },
  "run_result": {
    "status": "Finished",
    "execution_time": 0.002761363983154297,
    "return_code": 0,
    "stdout": "Hello, world!\\n",
    "stderr": ""
  },
  "executor_pod_name": null,
  "files": {}
}
```

## Datasets

<aside>

> 💡 Tip:
> You can find a simple playground page at [http://localhost:8080/SandboxFusion/playground/datasets](http://localhost:8080/SandboxFusion/playground/datasets)

</aside>

Sandbox Fusion integrates several dataset types including HumanEval, AutoEval, and CommonOJ, each with its own data format and evaluation method. Users interact with these datasets through a Python SDK that offers functions for code execution and evaluation: `run_code`, `get_prompts`, and `submit`. The SDK supports concurrent requests and allows API endpoint configuration through environment variables or functions. To evaluate code using a dataset, you first load it with the `load_dataset` function, configure the necessary settings, and use the `submit` function. For detailed instructions, see the [Sandbox Fusion documentation](https://bytedance.github.io/SandboxFusion/docs/docs/how-to/use-dataset/).

The Datasets module provides a unified interface for evaluating code across various datasets. Let's look at how to evaluate model outputs using the MBPP dataset as an example:

Get prompts for all MBPP questions:

```bash
curl 'http://localhost:8080/get_prompts' \
  -H 'Content-Type: application/json' \
  --data-raw '{"dataset":"mbpp","config":{}}'
```

Output:

```json
[
  {
    "id": 11,
    "prompt": "Write a python function to remove first and last occurrence of a given character from the string. Your code should satisfy these tests:\n\nassert remove_Occ(\"hello\",\"l\") == \"heo\"",
    "labels": {
      "challenge_test_list": [
        "assert remove_Occ(\"hellolloll\",\"l\") == \"helollol\"",
        "assert remove_Occ(\"\",\"l\") == \"\""
      ],
      "test_setup_code": ""
    }
  },
  {
    "id": 12,
    "prompt": "Write a function to sort a given matrix in ascending order according to the sum of its rows. Your code should satisfy these tests:\n\nassert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
    "labels": {
      "challenge_test_list": [],
      "test_setup_code": ""
    }
  },
  ...
]
```

Submit the model's output to get the **correctness result** for the problem:

```bash
curl 'http://localhost:8080/submit' \
  -H 'Content-Type: application/json' \
  --data-raw '{"dataset":"mbpp","id":"11","completion":"Here is a Python function that removes the first and last occurrence of a given character from a string:\n\n```python\ndef remove_Occ(s, char):\n    first_occ = s.find(char)\n    last_occ = s.rfind(char)\n    \n    if first_occ == -1 or first_occ == last_occ:\n        return s\n    \n    # Remove the first occurrence\n    s = s[:first_occ] + s[first_occ + 1:]\n    \n    # Adjust the index for the last occurrence since the string is now one character shorter\n    last_occ -= 1\n    \n    # Remove the last occurrence\n    s = s[:last_occ] + s[last_occ + 1:]\n    \n    return s\n\n# Test the function\nassert remove_Occ(\"hello\", \"l\") == \"heo\"\n```\n\nThis function works as follows:\n1. It finds the index of the first occurrence of the given character.\n2. It finds the index of the last occurrence of the given character.\n3. If the character does not exist in the string or only occurs once, it simply returns the original string.\n4. Otherwise, it constructs a new string by removing the first occurrence and then adjusts the index for the last occurrence before removing it.\n\nYou can run the provided test to ensure the function works as expected.","config":{}}'
```

Output:

```json
{
  "id": "11",
  "accepted": true,
  "extracted_code": "def remove_Occ(s, char):\n    first_occ = s.find(char)\n    last_occ = s.rfind(char)\n    \n    if first_occ == -1 or first_occ == last_occ:\n        return s\n    \n    # Remove the first occurrence\n    s = s[:first_occ] + s[first_occ + 1:]\n    \n    # Adjust the index for the last occurrence since the string is now one character shorter\n    last_occ -= 1\n    \n    # Remove the last occurrence\n    s = s[:last_occ] + s[last_occ + 1:]\n    \n    return s\n\n# Test the function\nassert remove_Occ(\"hello\", \"l\") == \"heo\"",
  "full_code": "def remove_Occ(s, char):\n    first_occ = s.find(char)\n    last_occ = s.rfind(char)\n    \n    if first_occ == -1 or first_occ == last_occ:\n        return s\n    \n    # Remove the first occurrence\n    s = s[:first_occ] + s[first_occ + 1:]\n    \n    # Adjust the index for the last occurrence since the string is now one character shorter\n    last_occ -= 1\n    \n    # Remove the last occurrence\n    s = s[:last_occ] + s[last_occ + 1:]\n    \n    return s\n\n# Test the function\nassert remove_Occ(\"hello\", \"l\") == \"heo\"\n\nassert remove_Occ(\"hello\",\"l\") == \"heo\"\nassert remove_Occ(\"abcda\",\"a\") == \"bcd\"\nassert remove_Occ(\"PHP\",\"P\") == \"H\"",
  "test_code": null,
  "tests": [
    {
      "passed": true,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.017310619354248047,
          "return_code": 0,
          "stdout": "",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": null
    }
  ],
  "extracted_type": null,
  "extra": null
}
```

# SandboxFusion API Usage

---

## Dataset Management

### List Datasets

- **Endpoint**: `/list_datasets`
- **Description**: Lists all registered datasets.

### List IDs

- **Endpoint**: `/list_ids`
- **Description**: Lists all IDs within a specified dataset.

### Get Prompt By ID

- **Endpoint**: `/get_prompt_by_id`
- **Description**: Retrieves a single prompt using its ID and dataset information.

### Get Prompts

- **Endpoint**: `/get_prompts`
- **Description**: Retrieves all prompts from a dataset.

## Code Execution

### Run Code

- **Endpoint**: `/run_code`
- **Description**: Executes a single code block.
- **Parameters**: Language, timeout settings, input/output files.

### Run Jupyter

- **Endpoint**: `/run_jupyter`
- **Description**: Executes multiple code cells within a Jupyter notebook environment.

## Evaluation

### Submit

- **Endpoint**: `/submit`
- **Description**: Submits a single problem's solution within a dataset, receiving feedback on its correctness and execution details.

### Get Metrics

- **Endpoint**: `/get_metrics`
- **Description**: Retrieves aggregated metrics for a dataset.

### Get Metrics Function

- **Endpoint**: `/get_metrics_function`
- **Description**: Provides the function used to generate metrics.

## Additional Endpoints

### Ping

- **Endpoint**: `/v1/ping`
- **Description**: Serves as a health check for the API.

### Root

- **Endpoint**: `/`
- **Description**: Provides general information or documentation.

## Python SDK Usage

### Installation

```bash
pip install sandbox-fusion
```

### Configuring API Endpoint

```python
from sandbox_fusion import set_endpoint
set_endpoint("http://your-api-endpoint.com")
```

### Run Code Example

```python
from sandbox_fusion import run_code, RunCodeRequest
run_code(RunCodeRequest(code='print(123)', language='python'))
```

### Submit Example

```python
from sandbox_fusion import submit, SubmitRequest
submit(SubmitRequest(...))
```

### Concurrent Requests Example

```python
from sandbox_fusion import run_concurrent, run_code, RunCodeRequest
codes = [f'print({i})' for i in range(123, 456)]
results = run_concurrent(run_code, args=[[RunCodeRequest(code=c, language='python')] for c in codes])
```

# Q & A
---

Q：为什么不采用创建Session+每个请求执行一个Cell的方式，而是要每次执行全部Cell？

A：为了维持沙盒服务的无状态特性，降低维护和使用成本。 沙盒服务于离线场景，吞吐的重要性大于延迟。

Potential Improvement: 设计一个online sandbox来服务Server-based Multi-turn rollout