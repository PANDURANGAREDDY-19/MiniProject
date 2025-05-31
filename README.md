# Machine Learning Project for Summer Vacation 
 The Goal of the Project is to identify the hidden trends between different features in the Dataset and predict or identify the possibility of heart diseases

## Attribute Information
=> Age: age of the patient [years] 

=> Sex: sex of the patient [M: Male, F: Female] 

=> ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]

=> RestingBP: resting blood pressure [mm Hg]

=> Cholesterol: serum cholesterol [mm/dl]

=> FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]

=> RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]

=> MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]

=> ExerciseAngina: exercise-induced angina [Y: Yes, N: No]

=> Oldpeak: oldpeak = ST [Numeric value measured in depression]

=> ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

=> HeartDisease: output class [1: heart disease, 0: Normal]

### Prerequisites
~ Anaconda prompt for Setting up your python environment
~ To create your environment manually follow the following steps:
    -> Open anaconda prompt
    -> Go to your Project directory
    -> type the following code
        conda create -p name_of_your_environment python=3.13 -y { You can create with your python version}
~ To activate your environment go to vs code and open your Project folder
    -Then open Terminal and open the Command Prompt
    -type the following code to activate your environment
        -> conda activate name_of_your_environment/
~ Python installed with version 3.8 or higher for development
~ pip installed in the Device for installing required libraries
~ Install the required libraries in the environment by using the following command
    ->pip install -r requirements.txt
  or install teh packages manually by checking with the requirements.txt file for packages/libraries
~ 