# Results and Release Notes

#### Results on test dataset (on 20% labeled): 
- Version 4 (Gray-ViT): 0.8194 (FFT), 0.9210 (Gray-ViT)  
- Version 4 (FFT-ViT): 0.8105 (FFT-ViT), 0.9218 (Gray)
- Version 3.5:  0.8222 (FFT), 0.9089 (Gray)
- Version 3:  0.8217 (FFT), 0.9173 (Gray)
- Version 2:  0.8137 (FFT), 0.9121 (Gray)
- Version 1:  0.8012 (FFT), 0.8746 (Gray)
- Independently trained branches: 0.7589 (FFT), 0.8907 (Gray)

#### Results on test dataset (on 50% labeled): 
- Version 4 (Gray-ViT): (FFT), (Gray-ViT)  
- Version 4 (FFT-ViT): (FFT-ViT), (Gray)
- Version 3.5: 0.8522 (FFT), 0.9328 (Gray)
- Version 1: 0.8452 (FFT), 0.9331 (Gray)
- Independently trained branches: 0.7891 (FFT), 0.8935 (Gray)

#### Results on test dataset (on 80% labeled): 
- Version 4 (Gray-ViT): (FFT), (Gray-ViT)  
- Version 4 (FFT-ViT): (FFT-ViT), (Gray)
- Version 3.5: 0.8522 (FFT), 0.9572 (Gray)
- Version 1: 0.8242 (FFT), 0.9423 (Gray)
- Independently trained branches: 0.8218 (FFT), 0.9068 (Gray)

#### Version 3.5 release notes
1. Integrated a weighted loss based pseudo-labeling.
#### Version 3 release notes
1. Integrated stochastic pseudo-labeled sample filtering.
2. Added MLFlow API for training and results visualization. 
#### Version 2 release notes
This version contains improvements for the cotraining algortihm after doing some research.
1. Changed the Loss function to contain the KL divergence on the FFT model.
2. The datasets will now always be the same for both models.
3. The pseudo-labeling is done together, based on the "opinion" of both models.
4. The confidence threshold is capped at the maximum value of 98.5%.
5. Reevaluation is done based on the agreement between both models.
#### Version 1 release notes
This version contains the prototype for the cotraining algortihm presented by Blum and Mitchell [1998].
Added features:
1. Added the backbone for SqueezeNet for the multi-branch neural network that contains one branch view for Grayscale images and one view for Frequency domain images. The FFT images represent only the **abosolute value of magnitude**.
2. The Loss function is a basic **sum of the two Cross Entropy results for each view.**
3. Each model will perform pseudo-label generation and will feed the Dataloader for the other model. The label is chosen by each model **independently** and inserted in the Dataloader of the other model.
4. Reevaluation of the added examples is done by feeding the samples to the models they were attributed to. **If the prediction is different**, the samples are removed from the datasets.
5. Confidence threshold is what determines a label to be chosen. This is set as a **fixed value** and is being changed based on the **removal rate during the reevaluation**.
6. Added a Learning Rate Scheduler.











