from pcse.db import NASAPowerWeatherDataProvider
from datetime import datetime
from fastai.utils.mod_display import *
from fastai.callbacks import *
from fastai.tabular import *
from fastai import *
import numpy as np
import pandas as pd
import os
from scipy import stats
from pcse.util import penman_monteith


def get_openweather_past5days(latitude, longitude, api_key):
    """AI is creating summary for get_openweather_past5days

    Args:
        latitude ([type]): [description]
        longitude ([type]): [description]
        api_key ([type]): [description]

    Returns:
        [type]: [description]
    """
    #https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&appid={API key}
    df_past = pd.DataFrame()

    for i in range(1,6):
        r = requests.get('https://api.openweathermap.org/data/2.5/onecall/timemachine?lat='+str(latitude)+'&lon='+str(longitude)+'&dt='+str(int(datetime.today().timestamp()-(i*24*3600)))+'&appid='+api_key+'&units=metric')
        df_past = df_past.append(pd.DataFrame(r.json()["hourly"]))

    df_past["dt"]=df_past.apply(lambda x: datetime.fromtimestamp(x["dt"]).date(),axis=1)

    df_past_2 = pd.DataFrame()
    df_past_2["TIMESTAMP_DAY"] = df_past.groupby("dt").mean().reset_index()["dt"]
    df_past_2["WIND"] = df_past.groupby("dt").mean().reset_index()["wind_speed"]
    df_past_2["TEMP"] = df_past.groupby("dt").mean().reset_index()["temp"]
    df_past_2["TMIN"] = df_past.groupby("dt").min().reset_index()["temp"]
    df_past_2["TMAX"] = df_past.groupby("dt").max().reset_index()["temp"]
    df_past_2["VAP"] = df_past.groupby("dt").mean().reset_index()["dew_point"]
    df_past_2["VAP"] = 6.11 * 10**((7.5 * df_past_2["VAP"])/(237.3 + df_past_2["VAP"]))

    try:
        df_past_2["RAIN"] = df_past.groupby("dt").sum().reset_index()["rain"]
    except:
        df_past_2["RAIN"]=0

    df_past_2["type"] = "past"
    
    return df_past_2


def get_openweather_7daysforecast(latitude, longitude, api_key):
    """AI is creating summary for get_openweather_7daysforecast

    Args:
        latitude ([type]): [description]
        longitude ([type]): [description]
        api_key ([type]): [description]

    Returns:
        [type]: [description]
    """

    # requesting 7-day forecast data from OpenWeatherMap API
    r = requests.get('https://api.openweathermap.org/data/2.5/onecall?lat='+str(latitude) \
                     +'&lon='+str(longitude)+'&exclude={part}&appid='+api_key+'&units=metric')

    # formatting the JSON response as Pandas dataframe
    weather_forecast = pd.DataFrame(r.json()["daily"])

    # renaming columns following the PCSE convention
    weather_forecast["TEMP"] = weather_forecast.apply(lambda x: x["temp"]["day"],axis=1)
    weather_forecast["TMIN"] = weather_forecast.apply(lambda x: x["temp"]["min"],axis=1)
    weather_forecast["TMAX"] = weather_forecast.apply(lambda x: x["temp"]["max"],axis=1)

    # actual vapour presure calculation ; in PCSE VAP stands for actual vapour pressure
    # formula from https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
    weather_forecast["VAP"] = 6.11 * 10**((7.5 * weather_forecast["dew_point"])/(237.3 + weather_forecast["dew_point"]))

    # renaming WIND and RAIN columns
    weather_forecast = weather_forecast.rename(columns={"wind_speed": "WIND"})

    if "rain" in weather_forecast :
        weather_forecast = weather_forecast.rename(columns={"rain": "RAIN"})
    else:
        weather_forecast["RAIN"]=0

    # converting timestamp formats
    weather_forecast["TIMESTAMP_DAY"] = weather_forecast.apply(lambda x: datetime.fromtimestamp(x["dt"]).date(),axis=1)
    weather_forecast["RAIN"] = weather_forecast["RAIN"].replace(np.nan,0)

    # cleaning    
    weather_forecast = weather_forecast[["TIMESTAMP_DAY","WIND","TEMP","TMIN","TMAX","VAP","RAIN"]]
    weather_forecast["type"]="forecast"
    
    return weather_forecast


def modelName_from_coordinates(model_class,latitude,longitude):
    """AI is creating summary for modelName_from_coordinates

    Args:
        model_class ([type]): [description]
        latitude ([type]): [description]
        longitude ([type]): [description]

    Returns:
        [type]: [description]
    """
    return model_class+"_"+str(round(latitude * 2) / 2)+"_"+str(round(longitude * 2) / 2)


def train_ANN(model_class, latitude, longitude, force_training=False, pctTest=0.01, loops=30, epochs=30, path="../data/"):
    """AI is creating summary for train_ANN

    Args:
        model_class ([type]): [description]
        latitude ([type]): [description]
        longitude ([type]): [description]
        force_training (bool, optional): [description]. Defaults to False.
        pctTest (float, optional): [description]. Defaults to 0.01.
        loops (int, optional): [description]. Defaults to 30.
        epochs (int, optional): [description]. Defaults to 30.
        path (str, optional): [description]. Defaults to "../data/".

    Returns:
        [type]: [description]
    """
    
    ### on vérifie si des modèles ont déjà été entraînés
    modelName = modelName_from_coordinates(model_class,latitude,longitude)
    trained_models = [name for name in os.listdir(path+"/models/") if os.path.isdir(path+"/models/"+name)]
    
    if (modelName in trained_models) and (force_training == False) :
        print("model already trained ; skipping training")
        return pd.DataFrame()
    else:
        # on déclanche l'entraînement
        force_training = True
    
    ### entraînement
    if force_training == True : 
        
        #### on récupère les données NASA Power
        print("Récupération des données NASA Power...")
        weatherdata = NASAPowerWeatherDataProvider(longitude,latitude,force_update=True) 
        print(weatherdata)
        
        df = pd.DataFrame(weatherdata.export())
        # converting PCSE units : cm to mm
        df["ET0"]=df["ET0"]*10 
        
        
        
        ### training parameters
        # paramètres et variables
        variableName="ET0" # variable à prédire
        df["id"]=df.index
        y_range=(0.9*min(df[variableName]) ,1.1*max(df[variableName])) # intervalle des valeurs à prédire
        layers=[6] # architecture du réseau
        ps=[0.5] # node dropout frequency
        emb_drop=0.5 # embedding dropout frequency
        bs=1024 # batch size

        cat_names=[]

        model_dict = {"M1":["TMIN","TMAX","RAIN","TEMP"],
                     "M2":["TMIN","TMAX","RAIN","TEMP","VAP"],
                     "M3":["TMIN","TMAX","RAIN","TEMP","WIND"],
                     "M4":["TMIN","TMAX","RAIN","TEMP","WIND","VAP"],
                     "M5":["TMIN","TMAX","RAIN","TEMP","WIND","VAP","IRRAD"],
                     "M6":["TMIN","TMAX","TEMP","IRRAD"],
                     "M7":["TMIN","TMAX","TEMP","WIND"],
                     "M8":["TMIN","TMAX","TEMP","RAIN","IRRAD"]}

        cont_names = model_dict[model_class]

        # définition du format du tableau de métriques
        results=pd.DataFrame(index=range(loops),columns=range(5)) 
        results.columns=["training R2","training RMSE","test R2", "test RMSE", "test stdev"]


        ### training loop
        for i in range(0,loops):
            print("")
            print("*** Entraînement du modèle "+model_class+" "+str(i)+" sur "+str(loops)+" : "+modelName+"_"+str(i)+" ***")
            print("")

            # on définit le nombre d'échantillons d'après le pourcentage de test split que l'on souhaite
            lenTest=int(round(pctTest*len(df.sample(frac=1).reset_index(drop=True)),0))

            # on sélectionne le test set
            testDf=df.sample(frac=1).reset_index(drop=True).iloc[0:lenTest]

            # on sélectionne le training/validation set en retirant les id d'échantillons qui correspondent au test set
            training=df[-df["id"].isin(testDf["id"])]

            # on définit le datablock fastai
            dep_var = [variableName] 
            procs = [FillMissing, Categorify, Normalize]

            # fastai nécessite de définir un test set même si l'on ne l'utilise pas, et split le training/validation set (variable "data") aléatoirement à 20% par défaut ("split_by_rand")
            test = TabularList.from_df(training.iloc[:].copy(), path=path, cont_names=cont_names)
            data = (TabularList.from_df(training.iloc[:], path=path, cont_names=cont_names, procs=procs)
                                   .split_by_rand_pct(0.2)
                                   .label_from_df(cols=dep_var)
                                   .add_test(test, label=0)
                                   .databunch(num_workers=0,bs=bs))

            # on déclare la topologie du NN, on lance la détermination du learning rate optimal et on lance l'apprentissage
            learn = tabular_learner(data, layers=layers, metrics=[rmse,r2_score], ps=ps, emb_drop=emb_drop, y_range=y_range)

            with progress_disabled_ctx(learn) as learn:
                learn.lr_find()
            learn.recorder.plot(suggestion=True)

            # décommenter la ligne suivante et indenter celle d'après pour ne pas afficher le suivi de l'entraînement en mode html
            # with progress_disabled_ctx(learn) as learn:
            learn.fit_one_cycle(epochs,learn.recorder.min_grad_lr,callbacks=[SaveModelCallback(learn,monitor='r2_score',mode='max',name=modelName+"-"+str(i))])
            #learn.fit_one_cycle(epochs,learn.recorder.min_grad_lr,callbacks=[SaveModelCallback(learn,monitor='root_mean_squared_error',mode='min',name=modelName+"-"+str(i))])

            try:
                os.mkdir(path+"/models/"+modelName)
            except:
                print("folder already exists")

            try:
                os.mkdir(path+"/models/"+modelName+"/"+modelName+"_"+str(i))
            except:
                pass
            learn.export(path+"/models/"+modelName+"/"+modelName+"_"+str(i)+"/export.pkl")

            # pour effectuer des prédictions sur le test set, on doit redéfinir un modèle d'apprentissage fastai
            test2 = TabularList.from_df(testDf.copy(), path=path, cont_names=cont_names, cat_names=cat_names)
            data2 = (TabularList.from_df(testDf, path=path, cont_names=cont_names, procs=procs)
                                   .split_by_rand_pct(0.2)
                                   .label_from_df(cols=dep_var)
                                   .add_test(test2, label=0)
                                   .databunch(num_workers=0))
            learn2 = tabular_learner(data2, layers=layers, metrics=[rmse,r2_score], ps=ps, emb_drop=emb_drop, y_range=y_range)

            # on charge les coefficients que l'on vient d'apprendre
            learn2.load(modelName+"-"+str(i))
            print("loading done")

            # on réalise la prédiction sur le test set
            test_predictions = learn2.get_preds(ds_type=DatasetType.Test)[0]
            test_predictions = [i[0] for i in test_predictions.tolist()]
            test_predictions = pd.DataFrame(test_predictions, columns = [variableName])

            # on réalise la régression des valeurs prédites pour le test set sur les valeurs ground truth afin de calculer le test R², RMSE et stdev
            X = testDf[variableName]
            Y = test_predictions

            dftest = pd.DataFrame(index=range(len(X)),columns=range(2))
            dftest.columns=["X","Y"]
            dftest["X"]=X
            dftest["Y"]=Y
            dftest2=pd.pivot_table(dftest, index='X', values='Y',aggfunc='mean')

            # on effectue le plot de la prédiction moyenne en fonction du truth
            X=dftest2.index
            Y=dftest2['Y']

            axes = plt.axes()
            axes.grid()
            plt.scatter(X,Y,s=5)

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
            def predict(x):
               return slope * x + intercept 
            fitLine = predict(X)
            plt.plot(X, fitLine, c='r')
            plt.plot(X, X, c='g')
            plt.show()

            error=X.copy()
            error=X-Y
            SE=pow(error,2)
            MSE=SE.mean()
            RMSE=math.sqrt(MSE)
            stdev=np.std(error)
            rsquared=np.corrcoef(X, Y)[0, 1]**2

            metrics=pd.DataFrame.from_records(learn.recorder.metrics)
            training_R2=max(metrics[1])
            training_RMSE=metrics[metrics[1]==training_R2][0]

            # on stocke les résultats de l'itération dans un tableau
            print("training R²=",float(training_R2),"training RMSE=",float(training_RMSE),"test R²=",rsquared,"test RMSE=",RMSE,"test stdev=",stdev)
            results.iloc[i,0]=float(training_R2)
            results.iloc[i,1]=float(training_RMSE)
            results.iloc[i,2]=rsquared
            results.iloc[i,3]=RMSE
            results.iloc[i,4]=stdev

        return results


def perform_ET0_prediction(model_class, latitude, longitude, dfPred, force_training=False, path='../data/', pctTest=0.01, loops=30, epochs=30):
    """AI is creating summary for perform_ET0_prediction

    Args:
        model_class ([type]): [description]
        latitude ([type]): [description]
        longitude ([type]): [description]
        dfPred ([type]): [description]
        force_training (bool, optional): [description]. Defaults to False.
        path (str, optional): [description]. Defaults to '../data/'.
        pctTest (float, optional): [description]. Defaults to 0.01.
        loops (int, optional): [description]. Defaults to 30.
        epochs (int, optional): [description]. Defaults to 30.

    Returns:
        [type]: [description]
    """
    modelName = modelName_from_coordinates(model_class,latitude,longitude)
    
    ### on vérifie si le modèle demandé a déjà été entraîné
    trained_models = [name for name in os.listdir(path+"/models/") if os.path.isdir(path+"/models/"+name)]
    
    if (modelName in trained_models) and (force_training == False) :
        print("model already trained ; skipping training")
    else:
        # on déclanche l'entraînement
        print("model "+modelName+" is not trained ; performing training")
        train_ANN(model_class, latitude, longitude, force_training=True)
    
    ### prediction
    # on détermine le nombre de modèles entraîné pour la combinaison coordonnées x classe
    modelNumbers = len([name for name in os.listdir(path+"/models/"+modelName) if os.path.isdir(path+"/models/"+modelName+"/"+name)])

    variableName="ET0" # variable à prédire
    print("")
    print("*** Modèle "+modelName+" ***")

    # on charge les données
    print("Loading dataframe...")
    dfPred[variableName] = 0

    model_dict = {"M1":["TMIN","TMAX","RAIN","TEMP"],
                 "M2":["TMIN","TMAX","RAIN","TEMP","VAP"],
                 "M3":["TMIN","TMAX","RAIN","TEMP","WIND"],
                 "M4":["TMIN","TMAX","RAIN","TEMP","WIND","VAP"],
                 "M5":["TMIN","TMAX","RAIN","TEMP","WIND","VAP","IRRAD"],
                 "M6":["TMIN","TMAX","TEMP","IRRAD"],
                 "M7":["TMIN","TMAX","TEMP","WIND"],
                 "M8":["TMIN","TMAX","TEMP","RAIN","IRRAD"]}

    cont_names = model_dict[model_class]
    cat_names = []

    for i in range(modelNumbers):
        print("Loading prediction model no "+str(i)+"...")
        procs = [FillMissing, Categorify, Normalize]
        data2 = TabularList.from_df(
            dfPred.copy(), path=path, cont_names=cont_names, procs=procs)
        learn2 = load_learner(path+"/models/"+modelName+"/"+modelName+"_"+str(i)+"/", test=data2)

        # on réalise la prédiction sur le test set
        print("Performing prediction...")
        test_predictions = learn2.get_preds(ds_type=DatasetType.Test)[0]
        test_predictions = [i[0] for i in test_predictions.tolist()]
        test_predictions = pd.DataFrame(test_predictions, columns=[variableName])
        dfPred[variableName+"-"+str(i)] = test_predictions

    print('Calculating ensemble prediction...')
    variable_cols = [col for col in dfPred.columns if variableName+"-" in col]
    dfPred[variableName] = dfPred[variable_cols].mean(axis=1) # median or mean
    dfPred[variableName+"_std"] = dfPred[variable_cols].std(axis=1)

    print("Done !")

    # calcul de l'IC90
    dfPred["IC90"]=1.645*dfPred["ET0_std"]/np.sqrt(modelNumbers)
    dfPred["incertitude"]=dfPred["ET0_std"]/dfPred["ET0"]

    return dfPred