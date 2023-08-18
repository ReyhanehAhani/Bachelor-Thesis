# Bachelor-Thesis
This repository contains 3 phases of my bachelor thesis project
# Road Sign Detection using CNN

## Contents

1. **Code 1: Basic CNN Approach**
   - File: `road_sign_detection_basic_cnn.ipynb`
   - This code implements a basic CNN architecture for road sign detection.

2. **Code 2: Data Augmentation and Ensemble**
   - File: `road_sign_detection_augmentation_ensemble.ipynb`
   - This code demonstrates data augmentation and ensemble techniques for improved accuracy.

3. **Code 3: Transfer Learning with MobileNet and Xception**
   - File: `road_sign_detection_transfer_learning.ipynb`
   - This code utilizes transfer learning with MobileNet and Xception architectures for road sign detection.

4. **Code 4: Advanced Ensemble with Multiple Models**
   - File: `road_sign_detection_advanced_ensemble.ipynb`
   - This code implements an advanced ensemble approach using multiple models for enhanced accuracy.

## Usage

Each code snippet is provided in a Jupyter Notebook format. You can run these notebooks using Jupyter or JupyterLab. The code snippets contain the following sections:

- Data preprocessing and augmentation
- Model architecture definition
- Model training and evaluation
- Visualization of training history and accuracy
- Confusion matrix visualization for evaluating results

## Dependencies

The following libraries are required to run the code snippets:

- TensorFlow
- Keras
- NumPy
- pandas
- matplotlib
- scikit-learn



# Lane and Drivable Area Detection Models

This repository contains code for implementing lane detection and drivable area detection using the U-Net architecture with attention mechanisms and the Focal Loss optimization. Below, you will find explanations of the model architectures and instructions for using the code snippets.

## Lane Detection using CULane and BDD Datasets

The lane detection code in this repository is designed to work with the CULane and BDD datasets. It employs the U-Net architecture with attention mechanisms and the Focal Loss for optimal lane detection.

### Features and Tasks

- **Dataset Preparation**: The `CULane` dataset is utilized for lane detection. Images and masks are preprocessed and loaded using a custom dataset class.

- **Data Augmentation**: Data augmentation techniques like rotation, flips, and dropout are applied to enhance training diversity.

- **Model Architecture**: The U-Net architecture is employed, which comprises encoding and decoding paths with attention mechanisms.

- **Loss Function**: The Focal Loss optimizes the model by focusing on challenging pixels.

- **Training and Validation**: The training loop, mixed-precision training, and validation process are implemented for optimal results.

### Instructions

1. **Dependencies**: Install required dependencies with `pip install pytorch_warmup torchview`.

2. **Dataset Preparation**: Update paths (`dataset_path`, `image_dir`, `mask_dir`) for the CULane dataset.

3. **Model Training**: Modify hyperparameters (`NUM_EPOCHS`, `BATCH_SIZE`, etc.) and run the training loop.

4. **Model Evaluation**: Evaluate the model on validation data and visualize predictions.

## Drivable Area Detection using Semantic Segmentation

The drivable area detection code is designed for the BDD100K dataset. It uses the U-Net architecture with attention mechanisms and the Focal Loss.

### Features and Tasks

- **Dataset Preparation**: The `BDD100K` dataset is used for drivable area detection.

- **Data Augmentation**: Similar data augmentation techniques are applied to the BDD100K dataset.

- **Model Architecture**: The U-Net architecture with attention mechanisms is employed for drivable area detection.

- **Loss Function**: The Focal Loss optimizes the model for accurate drivable area detection.

- **Training and Validation**: Training, mixed-precision training, validation, and evaluation processes are similar to the lane detection code.

### Instructions

1. **Dependencies**: Install required dependencies with `pip install pytorch_warmup torchview`.

2. **Dataset Preparation**: Update paths for the BDD100K dataset.

3. **Model Training**: Modify hyperparameters and train the model.

4. **Model Evaluation**: Evaluate the model on validation data and visualize predictions.

## Model Architecture Explanation

Both lane detection and drivable area detection models are based on the U-Net architecture with attention mechanisms. The U-Net architecture consists of encoding and decoding paths with skip connections and attention modules.

- **Encoding Path**: The encoding path includes convolutional blocks with downsampling.

- **Bottleneck Layer**: A bottleneck layer captures complex patterns in the encoded feature maps.

- **Decoding Path**: The decoding path uses up-convolutional blocks with attention mechanisms to upsample and refine feature maps.

- **Final Convolution**: The final convolutional layer produces predictions.

- **Focal Loss**: The Focal Loss function optimizes the model by focusing on challenging examples.

- **Mixed Precision Training**: Mixed-precision training accelerates training while maintaining stability.



# Reinforcement Learning Code

 Each code in the phase 3 snippet demonstrates the implementation and usage of a specific algorithm in the context of a given environment.

## Road Environment 

1. **Initialization and Parameters:**
   - The class constructor initializes parameters related to the road environment, such as road dimensions and the number of obstacles.
   - The `actions` dictionary maps action names to numeric values.
   - `state_features` list defines the state features, including car position, velocity, and obstacle positions.

2. **Resetting the Environment:**
   - The `reset` method prepares the environment for a new episode by generating obstacle positions and setting initial car state.

3. **State Transition and Step:**
   - The `transition` method simulates the car's velocity transition based on the chosen action.
   - The `step` method updates the car's position and calculates the reward based on physics and the reward function.

4. **Reward Function:**
   - The `reward_fn` method computes the reward based on the car's state and chosen action, promoting desired behaviors.

5. **Visualization (Tkinter):**
   - If `use_tkinter` is `True`, the environment's state is visualized using Tkinter.

6. **Root Update and Animation:**
   - The environment class uses Tkinter to create a visual animation of the car's movement and obstacle positions.

## Advantage Actor-Critic (A2C) Method 

This code implements the Advantage Actor-Critic (A2C) algorithm, combining actor and critic components for reinforcement learning.

1. **Neural Network Architecture (`DQN`):**
   - The `DQN` class defines a fully connected neural network model for Q-value estimation.

2. **Replay Memory (`ReplayMemory`):**
   - The `ReplayMemory` class implements a replay buffer for storing and sampling transitions.

3. **A2C Agent (`Agent`):**
   - The `Agent` class initializes the A2C agent with action space and state features.
   - It creates a policy network (`policy_net`) and a target network (`target_net`).

4. **Optimizing the Model:**
   - The `optimize_model` method updates the policy network using A2C by sampling transitions and performing backpropagation.

5. **Training Loop:**
   - The training loop iterates through episodes, interacting with the environment, and updating the policy network.

## SARSA (State-Action-Reward-State-Action) Method 

This code demonstrates the SARSA algorithm, an on-policy temporal difference learning algorithm.

1. **SARSA Agent (`SARSA_Agent`):**
   - The `SARSA_Agent` class initializes the SARSA agent with actions and learning rate.
   - It maintains a Q-table (`q_table`) for state-action values.

2. **Epsilon-Greedy Policy:**
   - The `select_action` method employs an epsilon-greedy strategy for action selection.

3. **SARSA Algorithm (`learn` Method):**
   - The `learn` method implements the SARSA algorithm, updating Q-values based on transitions.

4. **SARSA Update Rule:**
   - The SARSA update rule adjusts Q-values using the observed reward, next state, and next action.

## Proximal Policy Optimization (PPO) Method 

This code illustrates the Proximal Policy Optimization (PPO) algorithm, designed for stable policy optimization.

1. **PPO Agent (`PPO_Agent`):**
   - The `PPO_Agent` class initializes the PPO agent with actions, neural network, and hyperparameters.

2. **Policy and Value Networks:**
   - The policy network (`policy_net`) predicts action probabilities.
   - The value network (`value_net`) estimates state values.

3. **PPO Clipped Surrogate Objective:**
   - The `ppo_loss` method computes the PPO clipped surrogate objective for policy optimization.

4. **PPO Optimization (`optimize_model` Method):**
   - The `optimize_model` method updates the policy network using PPO and backpropagation.

5. **Training Loop:**
   - The training loop iterates through episodes, interacting with the environment, and optimizing the policy network using PPO.

6. **Value Function Training:**
   - The value network (`value_net`) is trained using mean squared error to predict state values.

PPO ensures stable updates and efficient handling of policy changes for reinforcement learning tasks.

