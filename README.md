# Project_Four: 

# How great is the weather risk for the CFA?

**The Problem:** 

The Country Fire Authority (CFA) is a volunteer-based organisation. It responds to a variety of different situations, including fire, flood, hazardous waste incidents, car crashes, and animal rescues.  The CFA are often first responders. 

The leader of the Ballarat brigade is very interested to understand if and how weather affects callout rates.  If it does, the CFA might be able to use weather forecasts to assist with their staffing and rostering planning.



**The Data:**

In order to analyse weather and its affects on callout rates, two openly available sources of data were used. 

The first source of data was the Bureau of Meteorology's daily weather observations for Ballarat.  This data series comes in monthly record subsets and was collected in two tranches.  First, from the Climate Data Online Site: (http://www.bom.gov.au/climate/data/?ref=ftr) but with further history accessed from the Trove repository, the National Library of Australia's digital repository.  Trove stores snapshots of selected websites over years. For example: (https://webarchive.nla.gov.au/awa/20230627014220/http://www.bom.gov.au/climate/dwo/). Data was collected for 2019-2022. 

A monthly weather record includes:

![](D:\Bootcamp\Classwork\Project_Four\BOM.png)

(Screen snapshot taken from: http://www.bom.gov.au/climate/dwo/202310/html/IDCJDW3005.202310.shtml)

The second source of information used is the CFA Incident Responses data (https://discover.data.vic.gov.au/dataset/cfa-incident-responses), available from the Data VIC website. This covers financial years 2004-2022.

The CFA data records:

| incident_datetime | incident_no | District_no | incident_type_code | incident_type |
| ----------------- | ----------- | ----------- | ------------------ | ------------- |


The daily weather records were downloaded in monthly view, month-by-month, from the Bureau's website and from Trove, and were accessible in csv format. Each individual financial year in the CFA incident sheet was saved as csv.

Originally, three weather stations were selected: Ballarat, Ferny Creek and Wangaratta, and three full years of data was selected  downloaded. Further records are available, but time constraints due to the manual nature of collecting the records (freely)  restrained the time period selected.



**Cleaning and processing:**

Data transformation and analysis was conducted in Python and Pandas, using Jupyter Notebook. 

The weather data required quite a bit of cleaning, including:

- Deleting columns that did not contain recorded values
- Removing NaN and np.nan, with 0 for some numeric and 'None' for categorical columns
- Interpolating some missing values where no value might affect the data
- Changing column types to datetime and numeric as required, for ease of processing in the system

- Dummy variables were put in place for wind direction, using OneHotEncoder.

For the CFA data, less cleaning was required.  The number of rows displaying NaN was small and not statistically relevant, so the empty rows were simply dropped using dropna(). 

Once the data was read into a dataset in Pandas dataframe, was cleaned and transformed, and combined with an inner join,  categorical columns were split into binary data using SciKit Learn OneHotEncoder, creating a unique column for each of the unique values of the original categorical columns. 



**Method** **& Machine Learning Selection**

An initial assessment of the data was undertaken in  Tableau: https://public.tableau.com/views/ProjectFourCFAAnalysisDashboard/WeatherVariablesandIncidentcountdashboard?:language=en-GB&:display_count=n&:origin=viz_share_link to identify the core components,  The focus was on the overall number of incident callouts by day. The whole state's data was used for the incident count, since although the data does include district numbers, the sheer number of callouts for mutual aid indicated that using the whole state's data would be more relevant and capture the effects of weather more adequately.



![](D:\Bootcamp\Classwork\Project_Four\Fig4.png)

There were significant amounts of dummy variables in the data once the categorical columns had been transformed, and the original intent to include three different weather stations as a snapshot of Victoria quickly showed to be causing issues with modelling the data. 

**Reduction of Variables**

Thus the CFA data was reduced to Ballarat, with incident_datetime and Incident_count, since for the purposes of predicting the volume of callouts (and thereby the need for more staffing in particular situations), the information about what type of callout it was was not relevant (even for animal rescue - which does still appear as a category for the CFA - but more likely for cows and sheep rather than black cats!).  

. The Ballarat weather data was reduced to:

```
Date                                      datetime64[ns]
Ball_Minimum temperature (°C)                    float64
Ball_Maximum temperature (°C)                    float64
Ball_Direction of maximum wind gust               object
Ball_Speed of maximum wind gust (km/h)           float64
```

The data was then combined in python using an inner join on Date/incident_date, and the data described for a quick overview, only of note was the Mean incident count of 104, and Mean wind gust of 45 km/h.  

![image-20231108142402484](C:\Users\rhian\AppData\Roaming\Typora\typora-user-images\image-20231108142402484.png)



A quick view of the incident count, maximum temperature and maximum wind gust shows similarities in underlying structures.  There are more call outs during summer when the temperature is high, for example.   

![](D:\Bootcamp\Classwork\Project_Four\Comb_1.png)



**Regression as the Machine Learning Model**

I concluded that a regression model with fewer variables would likely meet with some success, after a few false starts. I adopted a dual method approach, using both an Ordinary Least Squares model (OLS) from Statsmodels, and  SciKitLearn's Regression model. This combination was used to capture  which of the variables has impact on the outcome, trying to identify what the most significant weather aspects are that affect callouts for the CFA.

**Ordinary Least Squares model (Statsmodels): four iterations** 

The first iteration included only the Maximum temperature, with a constant. This first iteration is the basic 'baseline' for the OLS model. To rate the performance of the OLS model, I have focused on the coefficient of determination (R-squared) .

- The resultant R-squared value was 0.328

The second iteration included Minimum and Maximum temperature Celcius:

- The resultant R-squared value was 0.347, which is okay, but not the 0.8 target goal.  The coefficients are showing in the directions expected.  Higher maximum temperatures (fire risk) and higher winds were both causing more call outs. 

The third iteration had both Minimum and Maximum temperature Celcius, and included Max speed of wind gust (km/h)

- The Resultant R-squared value was 0.350, however, the coefficient for minimum temperature was negative. 

It seems plausible - particularly around Ballarat - that the minimum temperature is capturing call-outs for problems caused by low temperatures.  For example, low temperatures can cause frozen and burst water pipes.  

An inspection of the days with the highest volume of callouts showed something interesting: the most callouts were not the hottest days, and nor were they the windiest.  Instead, it was when both temperature and wind were high that the most callouts occurred.  

To increase the accuracy of the model, the cross product of the Maximum temperature and the Maximum speed of wind gust was included for the second iteration. 

- This raised the R-squared to **0.377**

The final iteration included minimum temperature, maximum wind gust, and with a cross product of the Max speed of wind gust and Max temperature.

- This alternative formulation lowered the R-squared to **0.374**

So, while the model does not perform to the target 0.8, it is amenable to some refinement to increase the score by 0.5.

It does show that there is correlation between weather and callout rates, and the increase with the cross product does seem to indicate that the combination of strong maximum wind gusts AND high temperatures has a multiplying effect to the callout rate.  Using the cross-product has shown the non-linearity of the problem - but we leave the analysis of exactly which relative proportions in the cross-product term to future work. 

**Linear Regression**

To test against another regression model, SciKitLearn's Linear Regression was also used against the model, **including** the cross product of the maximum temperature and maximum wind gust speed.

The model was split into training and testing, and the X-train data was scaled using Standard Scaler, and model fitted and run.  

![](D:\Bootcamp\Classwork\Project_Four\Fig2.png)

**Results**



While this shows that the data is not linear, it does show that the model achieves some moderate and useful accuracy in predicting callout rates.  There is one extreme case that the model managed to predict, in a manner of speaking.  The model suggested an extremely high number of callouts was likely.  The prediction was not as high as the actual callout numbers, but nonetheless, it was still an important leading indicator of a large volume of callouts. The model did not handle the second extreme case as well, and the number of callouts was quite under-predicted.

The residuals continue this story:

![](D:\Bootcamp\Classwork\Project_Four\Fig_3.png)

While a perfect fit for the residuals would be a bell-shape peaking at 0, this is almost, but not quite there. The two outer points again emphasise the need to explore what the extreme instances were that could account for it. August 2020 saw large storm damage in the central district, at a time when Fire services would not be expecting peak demand as is usual in the later summer months.



**Conclusion**

•While the linear regression was not the best predictor, and did not perform to the degree necessary, it has:

•- Highlighted that there is correlation between weather and callout rates

•- the combination of strong maximum wind gusts AND high temperatures has a **multiplying effect:** the risk of callouts is much greater with hot and windy weather. 

•-Using the cross-product has shown the non-linearity of the problem - the analysis of exactly which relative proportions in the cross-product term is left for future work. 



**Sources and Acknowledgements**

Daily observations are available from http://www.bom.gov.au/climate/data. Copyright Commonwealth of Australia, Bureau of Meteorology, and archives of the National Library of Victoria's Digital repository, TROVE, available at :  (https://webarchive.nla.gov.au, see specific examples:https://webarchive.nla.gov.au/awa/20230627014220). No data scraping techniques were used to collect this information: it was purely human mechanical actions. And many of them.

https://discover.data.vic.gov.au/dataset/cfa-incident-responses

A breakdown of different types of regression: https://www.analyticsvidhya.com/blog/2022/01/different-types-of-regression-models/

https://www.statsmodels.org/stable/index.html



