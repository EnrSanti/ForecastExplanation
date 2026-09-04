% Example generated data for day (2025, 9, 1)

#pos(e264@1000,{ 

forecasted_sky(barcis, "mostly_cloudy", autumn),
forecasted_rain(barcis, 0, autumn)},
{
sunny_at(barcis,autumn), 
partially_sunny_at(barcis,autumn), 
rains_at(barcis,1,autumn), 
rains_at(barcis,2,autumn), 
rains_at(barcis,4,autumn), 
rains_at(barcis,6,autumn)
},
{
location_considered(barcis). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 1).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_750m_covers(udine_palmanova,753, 8).
cloud_at_750m_covers(lignano_grado,753, 9).
cloud_at_750m_covers(udine_palmanova,752, 10).
cloud_at_750m_covers(pordenone,752, 10).
cloud_at_750m_covers(gemona_stolvizza,761, 23).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 1).
cloud_at_1_4km_covers(gemona_stolvizza,1075, 1).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 2).
cloud_at_1_4km_covers(gemona_stolvizza,1075, 2).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 3).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 4).
cloud_at_1_4km_covers(gemona_stolvizza,1075, 6).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 7).
cloud_at_1_4km_covers(gemona_stolvizza,1075, 7).
cloud_at_1_4km_covers(pontebba_tarvisio,1075, 8).
cloud_at_1_4km_covers(udine_palmanova,1082, 10).
cloud_at_1_4km_covers(gorizia,1083, 12).
cloud_at_1_4km_covers(trieste,1084, 12).
cloud_at_1_4km_covers(gemona_stolvizza,1082, 12).
cloud_at_1_4km_covers(sappada_forni_villa,1087, 18).
cloud_at_1_4km_covers(gemona_stolvizza,1087, 19).
cloud_at_1_4km_covers(gemona_stolvizza,1087, 20).
cloud_at_1_4km_covers(gemona_stolvizza,1087, 21).
cloud_at_1_4km_covers(sappada_forni_villa,1087, 22).
cloud_at_1_4km_covers(barcis,1087, 22).
cloud_at_1_4km_covers(gemona_stolvizza,1087, 22).
cloud_at_1_4km_covers(pordenone,1087, 22).
cloud_at_1_4km_covers(sappada_forni_villa,1087, 23).
cloud_at_1_4km_covers(barcis,1087, 23).
cloud_at_1_4km_covers(gemona_stolvizza,1087, 23).
cloud_at_3km_covers(sappada_forni_villa,1343, 22).
cloud_at_9km_covers(lignano_grado,1518, 7).
cloud_at_9km_covers(gorizia,1524, 12).
cloud_at_9km_covers(pordenone,1525, 12).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,279.13).
% temperature_at_afternoon(sappada_forni_villa,279.42).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.60).
% temperature_at_afternoon(pontebba_tarvisio,278.54).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,278.37).
% temperature_at_afternoon(lignano_grado,278.46).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,279.40).
% temperature_at_afternoon(barcis,279.38).
temperature_decreased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,278.53).
% temperature_at_afternoon(udine_palmanova,278.54).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,278.37).
% temperature_at_afternoon(gorizia,278.81).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,277.87).
% temperature_at_afternoon(trieste,278.31).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.73).
% temperature_at_afternoon(gemona_stolvizza,279.04).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,279.07).
% temperature_at_afternoon(pordenone,278.40).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,40.00).
% humidity_at_afternoon(sappada_forni_villa,26.67).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,41.33).
% humidity_at_afternoon(pontebba_tarvisio,43.75).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,44.67).
% humidity_at_afternoon(lignano_grado,64.17).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,29.33).
% humidity_at_afternoon(barcis,29.58).
humidity_increased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,39.33).
% humidity_at_afternoon(udine_palmanova,60.00).
humidity_increased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,40.00).
% humidity_at_afternoon(gorizia,66.67).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,52.00).
% humidity_at_afternoon(trieste,62.08).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,37.33).
% humidity_at_afternoon(gemona_stolvizza,36.67).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,38.67).
% humidity_at_afternoon(pordenone,52.50).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"NE",21).
wind_blowing_afternoon(sappada_forni_villa,"NE",27).
wind_blowing_morning(pontebba_tarvisio,"NE",19).
wind_blowing_afternoon(pontebba_tarvisio,"NE",26).
wind_blowing_morning(lignano_grado,"NE",21).
wind_blowing_afternoon(lignano_grado,"NE",27).
wind_blowing_morning(barcis,"NE",21).
wind_blowing_afternoon(barcis,"NE",27).
wind_blowing_morning(udine_palmanova,"NE",18).
wind_blowing_afternoon(udine_palmanova,"NE",24).
wind_blowing_morning(gorizia,"NE",18).
wind_blowing_afternoon(gorizia,"NE",24).
wind_blowing_morning(trieste,"NE",18).
wind_blowing_afternoon(trieste,"NE",24).
wind_blowing_morning(gemona_stolvizza,"NE",19).
wind_blowing_afternoon(gemona_stolvizza,"NE",26).
wind_blowing_morning(pordenone,"NE",21).
wind_blowing_afternoon(pordenone,"NE",27).

temp_front_morning_at_100m(lignano_grado,trieste).
temp_front_morning_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(lignano_grado,trieste).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(lignano_grado,udine_palmanova).
temp_front_morning_at_750m(lignano_grado,udine_palmanova).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,gorizia).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(pordenone,udine_palmanova).
temp_front_morning_at_1_4km(barcis,pordenone).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_1_4km(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_1_4km(pordenone,sappada_forni_villa).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_3km(barcis,sappada_forni_villa).
temp_front_morning_at_3km(pordenone,udine_palmanova).
temp_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_5_5km(gemona_stolvizza,sappada_forni_villa).

hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_100m(lignano_grado,trieste).
hum_front_morning_at_100m(gorizia,lignano_grado).
hum_front_morning_at_100m(gemona_stolvizza,gorizia).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_1_4km(gemona_stolvizza,gorizia).
hum_front_morning_at_3km(gorizia,lignano_grado).
hum_front_morning_at_3km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_3km(gorizia,lignano_grado).
hum_front_afternoon_at_5_5km(lignano_grado,trieste).
hum_front_morning_at_9km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_9km(pordenone,udine_palmanova).
hum_front_afternoon_at_9km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_9km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_9km(gorizia,lignano_grado).

}). 
