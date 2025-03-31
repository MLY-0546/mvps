# mvps
The setting of our experiment is using BatchSize 4 (~26G) and learning rate is 1e-4. We conduct our experiment on A100 or A6000. Trainning epoch is 5. The setting of experiment is important to the result.  We remain consistent with previous work. Due to the instability of mixed-precision training, the best-performing weights on the validation set can be manually selected for testing on the test set. Additionally, this code version has slight differences from the version in the paper. In this repository, we only provide the method. Since we are expanding our approach, the complete training and testing process will be made public after the expansion is completed.

Compared to the original method, we use an enhancement technique that scales and rotates the same image to generate four images, which are then combined into one. Additionally, we use the CUPY library to accelerate the model's testing phase.








