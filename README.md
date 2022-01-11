# NN-ET0: *estimating reference evapotranspiration through training of local neural network models from reanalysis climate datasets*

Estimation of reference evapotranspiration (ET0) is an important part of precision irrigation (PI) practices. ET0 can be estimated through the use of reference formulas such as FAO56 Penman-Montheith, however it requires a lot of weather variables to be measured. Access to these variables and to the needed equipment for accurate measurement is challenging, especially in developing countries. 

In this method, datasets from climatic assimilation models (AM) such as NASA Power or OpenWeatherMap are used to train local artificial neural network ET0 (ANN ET0) estimation models. We propose a Python software, NN-ET0, that automatically performs the training of a local ANN ET0 estimation model as well as ET0 estimation from a very simple weather forecast API, so that every irrigation practitioner can use it in their day-to-day work. 

An upcoming paper will display in details the method and associated performances.

## Usage
**Note : made on Python 3.9.6**

Once NN-ET0 is cloned, open the notebook in `notebooks/NN-ET0.ipynb`. 
The second cell will need to be edited as follows:
- if not already done, register at [OpenWeatherMap](https://openweathermap.org/) and subscribe to the free tier for 7-day forecast. Paste the API key in the notebook under the variable `api_key`. OpenWeatherMap indicates their maps have a 500m precision.

**Note**: other weather datasets (NasaPower, ERA5 ...) can be used as long as the input resembles a dataframe as in cell #5:

|   	| TIMESTAMP_DAY 	|   WIND   	|    TEMP   	|  TMIN 	|  TMAX 	|    VAP    	| RAIN 	|   type   	|
|:-:	|:-------------:	|:--------:	|:---------:	|:-----:	|:-----:	|:---------:	|:----:	|:--------:	|
| 0 	| 2021-10-02    	| 1.508333 	| 20.927917 	| 14.56 	| 28.02 	| 11.350648 	| 0.00 	| past     	|
| 1 	| 2021-10-03    	| 0.995417 	| 21.849167 	| 15.81 	| 27.81 	| 12.487504 	| 0.00 	| past     	|
| 2 	| 2021-10-04    	| 2.360000 	| 22.380000 	| 16.41 	| 23.61 	| 12.966167 	| 1.45 	| forecast 	|
| 3 	| 2021-10-05    	| 3.660000 	| 27.520000 	| 15.77 	| 27.52 	| 11.154750 	| 0.00 	| forecast 	|
| 4 	| 2021-10-06    	| 4.040000 	| 27.770000 	| 15.31 	| 29.20 	| 8.786952  	| 0.00 	| forecast 	|

- indicate the `latitude` and `longitude` of your area,
- chose the `model_class`, i.e. which model to use based on the following choices (M5 displaying the best results in our study) :

| model alias 	| features used for training               	|
|-------------	|------------------------------------------	|
| M1          	| TEMP, TMIN, TMAX, RAIN                   	|
| M2          	| TEMP, TMIN, TMAX, RAIN, VAP              	|
| M3          	| TEMP, TMIN, TMAX, RAIN, WIND             	|
| M4          	| TEMP, TMIN, TMAX, RAIN, VAP, WIND        	|
| M5          	| TEMP, TMIN, TMAX, RAIN, VAP, WIND, IRRAD 	|
| M6          	| TEMP, TMIN, TMAX, IRRAD                  	|
| M7          	| TEMP, TMIN, TMAX, IRRAD                  	|
| M8          	| TEMP, TMIN, TMAX, RAIN, IRRAD            	|

- `path` to save your data to.

Then, run all the cells of the notebook. It may take some time as several ANN models are trained (between 30 to 60 minutes for M5).

Outputs include:
- each trained model performances on past data,
- a temporal dataframe with the `ET0` variable, the `ET0-std`, `incertitude` of the value and all ANN outputs,
- a graphic of past ET0, as taken from NasaPower API, and estimated ET0.

---
