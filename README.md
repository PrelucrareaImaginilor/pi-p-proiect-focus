[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BzgEFjMi)

# Recunoașterea persoanelor după mers




## Descriere scurtă a proiectului

Proiectul urmărește dezvoltarea unei aplicații de **recunoaștere a persoanelor după mers**, folosind seturi de date publice și tehnici de **învățare profundă**.  
Arhitectura sistemului include etapele de achiziție, preprocesare, extragere și selecție a trăsăturilor, urmate de clasificare.  
Scopul este obținerea unui model capabil să identifice persoane în condiții variate (unghiuri, iluminare, zgomot de fundal), cu o acuratețe cât mai ridicată.


## Analiza literaturii de specialitate

| Nr. | Autor(i) / An                                                                                                        | Titlul articolului / proiectului                                                                           | Aplicație / Domeniu       | Tehnologii utilizate                                                       | Metodologie / Abordare                                                                                                                                                                                                                                 | Rezultate                           | Limitări                                                                                      | Comentarii suplimentare                                         |
| --- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | Xiaoyan Xie, Shaorun Yin, Yun Zhu, Huan Zhao, Panyu Cao, Miaomiao Chai, 2022                                         | **Gait Feature Extraction and Recognition Based on Video Compression**                                     | Identificarea persoanelor | NResCNN                                                                    | Analizarea zgomotului produs de mișcările siluetei umane                                                                                                                                                                                               | Rata de acuratețe medie: **97.75%** | Prezintă acuratețe scăzută la unghiuri apropiate de 0° sau 180°                               | —                                                               |
| 2   | Veenu Rani, Munish Kumar, 2023                                                                                       | **Human gait recognition: A systematic review**                                                            | Identificarea persoanelor | CCTV camera                                                                | CNN-based deep learning algorithm                                                                                                                                                                                                                      | **95% acuratețe**                   | Alegerea valorilor filtrelor (kernel-urilor) convoluționale                                   | —                                                               |
| 3   | Ch Avais Hanif, Muhammad Ali Mughal, Muhammad Attique Khan, Nouf Abdullah Almujally, Taerang Kim, Jae-Hyuk Cha, 2024 | **Human Gait Recognition for Biometrics Application Based on Deep Learning Fusion Assisted Framework**     | Identificarea persoanelor | 5 rețele neuronale (WNN, NNN, MNN, Bi-layered NN, Tri-layered NN) + MATLAB | Extragerea cadrelor, reglarea fină a modelelor CNN pre-antrenate folosind învățarea prin transfer, extragerea trăsăturilor profunde din ambele fluxuri, optimizarea trăsăturilor și fuziunea vectorilor de trăsături optimizați, urmată de clasificare | **94.14% acuratețe**                | Mărimea datasetului                                                                           | —                                                               |
| 4   | Muhammad Bilal, He Jianbiao, Husnain Mushtaq, Muhammad Asim, Gauhar Ali, Mohammed ElAffendi, 2024                    | **GaitSTAR: Spatial–Temporal Attention-Based Feature-Reweighting Architecture for Human Gait Recognition** | Identificarea persoanelor | Convolutional Neural Network (CNN)                                         | Se introduce o nouă metodă „GaitSTAR” care integrează caracteristicile spațiale cu datele temporale                                                                                                                                                    | **84.13% acuratețe medie**          | Cost mare de implementare                                                                     | —                                                               |
| 5   | Sk Aspak Ali, Gona Sai Charan, Sai Kalyan Tirumalasetty, Anand Singh, Dr. Dhiraj Kapila, 2024                        | **Human Gait Recognition using Machine Learning Technologies for Inclusive Innovation**                    | Identificarea persoanelor | Tehnici deep learning, OpenCV                                              | MoveNet este folosit ca rețea de extracție de trăsături și predicție a punctelor-cheie ale corpului uman. Rezultatele (coordonate numerice) sunt apoi introduse într-un algoritm de învățare automată clasic (ML) pentru clasificare.                  | —                                   | Videoclipurile au fps mare → frame-urile trebuie redimensionate la 192×191×3 pentru eficiență | Eliminarea fundalului nu îmbunătățește semnificativ performanța |






## Proiectarea soluției

### Schema bloc a sistemului

![workflow](https://github.com/user-attachments/assets/d3a11d7c-fc68-4796-8c3e-e90a86d367a4)


### Detalierea componentelor

1. Achiziția datelor - Se realizează utilizând seturi de date publice precum CASIA-B, CASIA-E, TUM-IITKGP Gait Database și OU-ISIR Biometric Database, care conțin secvențe video cu persoane surprinse în mișcare din diferite unghiuri.
2. Preprocesare - Include pași precum eliminarea zgomotului, stabilizarea cadrelor, normalizarea rezoluției, extragerea siluetelor și scăderea fundalului, pentru a izola mișcarea subiectului.
3. Feature Extraction - Folosirea Rețelelor Neuronale Convoluționale(CNN) (ex. MoveNet, OpenPose, Gait Energy Image – GEI) pentru identificarea trăsăturilor mersului.
4. Feature Selection - Se aplică metode de optimizare și reducere a dimensionalității (PCA, LDA) pentru a păstra doar caracteristicile relevante și a crește performanța clasificării.
5. Clasificare - Se utilizează algoritmi de învățare automată sau profundă (SVM, CNN, Bi-LSTM etc.) pentru a identifica persoana în funcție de tiparul mersului.
