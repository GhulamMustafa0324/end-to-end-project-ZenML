
# End-to-End MLflow Project with ZenML

This project demonstrates a complete end-to-end machine learning pipeline using ZenML and MLflow. The focus is on writing high-quality code, following coding conventions, applying Object-Oriented Programming (OOP) principles, and maintaining detailed logging throughout the project lifecycle.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, we implement a machine learning pipeline from data ingestion to model deployment. Key features include writing clean and maintainable code, modular design using OOP principles, and robust tracking and logging using ZenML and MLflow. By using ZenML for orchestration and MLflow for experiment tracking and model management, this project ensures reproducibility and scalability.

### Key Objectives:
1. Create a reusable and well-structured machine learning pipeline.
2. Leverage ZenML to orchestrate various stages of the pipeline.
3. Use MLflow to track experiments, monitor model performance, and handle versioning.
4. Ensure adherence to coding conventions and use OOP for modularity and readability.
5. Maintain comprehensive logging to debug and trace model performance.

## Features
- **ZenML Integration**: Orchestrates data pipelines and connects various stages.
- **MLflow Tracking**: Tracks and logs experiments, models, parameters, and results.
- **OOP Principles**: Ensures scalability, code reuse, and organization.
- **Logging with ZenML**: Enables debugging and traceability throughout the pipeline.
- **Modular Design**: Each step in the pipeline is modular, promoting flexibility and reuse.

## Tech Stack
- **Python**: Core programming language for data processing and model training.
- **ZenML**: Pipeline orchestration framework to manage and scale machine learning workflows.
- **MLflow**: Experiment tracking and model management.
- **scikit-learn**: Machine learning library for building models.
- **Pandas/Numpy**: Data manipulation libraries.
- **Matplotlib/Seaborn**: Visualization libraries for data analysis.

## Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/GhulamMustafa0324/end-to-end-project-ZenML.git
   cd end-to-end-project-ZenML
   \`\`\`

2. Create and activate a virtual environment:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate  # For Windows
   \`\`\`

3. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Initialize ZenML:
   \`\`\`bash
   zenml init
   \`\`\`

5. Install MLflow:
   \`\`\`bash
   pip install mlflow
   \`\`\`

## Project Structure
\`\`\`
end-to-end-project-ZenML/
│
├── data/                       # Raw and processed data files
├── logs/                       # Logs generated during pipeline runs
├── models/                     # Stored machine learning models
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── src/                        # Source code for the project
│   ├── pipelines/              # ZenML pipeline definitions
│   ├── steps/                  # Individual steps for data processing, training, etc.
│   ├── utils/                  # Utility functions (e.g., logging, data loading)
│
├── mlruns/                     # MLflow experiment tracking
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── zenml.yaml                  # ZenML project configuration
\`\`\`

## Usage

### 1. Data Preparation
Make sure the data is in the `data/` directory. You can customize data loading in the `src/steps/data_loader.py` file.

### 2. Running the Pipeline
To run the pipeline, follow these steps:

\`\`\`bash
# Step 1: Train the model and track experiments
zenml pipeline run train_pipeline

# Step 2: Check results in MLflow UI
mlflow ui
\`\`\`

### 3. Experiment Tracking
MLflow tracks all the experiments, metrics, parameters, and artifacts. Access the MLflow UI to monitor results:

\`\`\`bash
mlflow ui
\`\`\`

Go to `http://localhost:5000` to access the UI.

## Results

- The pipeline logs all intermediate steps and metrics to MLflow, allowing you to visualize the performance of different experiments.
- Detailed logs are available in the `logs/` folder for further analysis.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request. We welcome contributions that improve code quality, add new features, or enhance the current pipeline.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
