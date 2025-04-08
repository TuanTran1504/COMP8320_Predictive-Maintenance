In this study, I carried out experiments to evaluate different machine learning models for 
predictive maintenance (PdM), focusing on forecasting potential failures in industrial 
equipment and estimating their remaining useful life (RUL). Specifically, I will perform two 
experiments using two distinct detection categories to validate the findings of this research. 
The models were trained using historical sensor data from the UCI AI4I 2020 Predictive 
Maintenance Dataset and Unbalance Detection of a Rotating Shaft Using Vibration Data with 
the goal of predicting failure occurrences or forecasting the health degradation of assets. With 
this data we can train model to detect anomalies as well as unbalance detection on rotating 
shaft 

**Anomaly Detection using One-class Autoencoder **  

An autoencoder, which is a type of neural network, is designed to encode input data into a 
smaller dimensional form and then decode it back to its original format. The training process 
of One-class Autoencoder involves only non-failure or "normal" operational data. The 
autoencoder can learn the patterns of this normal data very efficiently. When it encounters 
data that resembles what it was trained on, it reconstructs it with low error. Conversely, if the 
autoencoder processes data that deviates from the norm, such as indications of a failure or 
anomaly, its reconstruction error increases significantly. This occurs because the model hasn't 
been trained to recognize and reconstruct these abnormal patterns. However, if the model 
fails to effectively learn the patterns of normal data, it may incorrectly classify normal 
instances as anomalies. Additionally, when new types of normal data are introduced, the 
model might misidentify them as anomalies if it lacks good generalization capability. 

<img width="373" alt="image" src="https://github.com/user-attachments/assets/686bf4b4-b0d4-4e83-ae2f-ae825e76aafd" />

<img width="380" alt="image" src="https://github.com/user-attachments/assets/0e030f9f-5fbf-478d-9471-a72c25db146e" />

**Unbalance Detection using MLP**  

The dataset comprises vibration readings from three sensors mounted on a rotating shaft, 
along with five distinct levels of unbalance introduced by adding a weight to one side. 
However, due to the massive volume of data—25 million entries—the model struggled to 
effectively capture the underlying patterns and correctly classify the labels (77% accuracy). 
This situation underscores the challenge of applying such techniques in real-life scenarios 
with enormous datasets, where substantial resources are required. Utilizing cloud computing 
could greatly enhance the ability to process and analyze such large-scale data more 
efficiently.

<img width="215" alt="image" src="https://github.com/user-attachments/assets/adddb047-a445-4bac-8f3a-314ef413beb0" />


<img width="473" alt="image" src="https://github.com/user-attachments/assets/21b3a47b-349d-4a27-aa3a-30eb71fe8b22" />
