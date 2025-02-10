# RoboBERT
This is the official implementation of RoboBERT, which is a novel end-to-end multiple-modality robotic operations training framework. It allows an instruction-following agent, switching the policy by natural language input. The model itself is an extension or improvement of the traditional CNN-based diffusion policy, adding some creative training methods. For examples, two-stage training separates learning of different modalities; data-augmentation reduces the reliance of extra data, etc. The training framework has been tested on changllenging CALVIN benchmark and get the SOTA comparing to models without using large pretraining or extra dataset on ABCD -> D. Additionally, it also only use the trajectories with language labels. Based on these reasons, the model is much more easily to train. The repository shows the training/testing source code for repeative experiments. If you want to learn more about this work, please check our paper. Thank you!

## Model Structure

## Usage
### Downloading Dataset and Configurating Environment
Please downloading the necessary dataset and preparing the corresponding environment on [CALVIN](https://github.com/mees/calvin). Besides, [BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main) is also needed in this project. Although the project and related libraries have been confirmed to run successfully on Windows by us, it is found that some libraries like Pyhash is difficult to compile and some performance loss for the model may also occurr on Windows, Linux is strongly recommended.

### Separating the Actions from Dataset
Because the training reads actions data, which is not very large in total, more frequently than image data, it can reduce the I/O frequency and extend the harddisk life if all the action data is restored in RAM in one time. We have created a script extracting the actions data individually from original CALVIN dataset and restoring into a pkl file. During the training, the RGB observations and actions are read from original CALVIN and extracted file respectively.  

### Training the Model
After completing the two steps above, 

### Evaluating the Model


