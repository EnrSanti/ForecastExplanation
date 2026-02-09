% Example generated data for day (2025, 9, 4)

#pos(e134@1000,{ 

forecasted_sky(pontebba_tarvisio, "mostly_clear", autumn),
forecasted_rain(pontebba_tarvisio, 0, autumn)},
{
partially_sunny_at(pontebba_tarvisio,autumn), 
covered_at(pontebba_tarvisio,autumn), 
rains_at(pontebba_tarvisio,1,autumn), 
rains_at(pontebba_tarvisio,2,autumn), 
rains_at(pontebba_tarvisio,4,autumn), 
rains_at(pontebba_tarvisio,6,autumn)
},
{
location_considered(pontebba_tarvisio). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 4).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_9km_covers(barcis,1615, 15).
cloud_at_9km_covers(sappada_forni_villa,1614, 16).
cloud_at_9km_covers(barcis,1614, 16).
cloud_at_9km_covers(gemona_stolvizza,1614, 17).
cloud_at_9km_covers(pontebba_tarvisio,1618, 18).
cloud_at_9km_covers(lignano_grado,1618, 18).
cloud_at_9km_covers(udine_palmanova,1618, 18).
cloud_at_9km_covers(gemona_stolvizza,1618, 18).
cloud_at_9km_covers(pontebba_tarvisio,1614, 19).
cloud_at_9km_covers(gorizia,1614, 19).
cloud_at_9km_covers(gemona_stolvizza,1614, 19).
cloud_at_9km_covers(pontebba_tarvisio,1614, 20).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,272.37).
% temperature_at_afternoon(sappada_forni_villa,277.94).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,276.23).
% temperature_at_afternoon(pontebba_tarvisio,277.50).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.23).
% temperature_at_afternoon(lignano_grado,277.98).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,274.27).
% temperature_at_afternoon(barcis,278.58).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,275.97).
% temperature_at_afternoon(udine_palmanova,277.15).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,275.80).
% temperature_at_afternoon(gorizia,277.77).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,275.83).
% temperature_at_afternoon(trieste,278.33).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,274.73).
% temperature_at_afternoon(gemona_stolvizza,278.58).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.47).
% temperature_at_afternoon(pordenone,277.52).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,62.00).
% humidity_at_afternoon(sappada_forni_villa,39.17).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,48.00).
% humidity_at_afternoon(pontebba_tarvisio,31.67).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,60.67).
% humidity_at_afternoon(lignano_grado,62.92).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,50.67).
% humidity_at_afternoon(barcis,29.58).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,63.33).
% humidity_at_afternoon(udine_palmanova,56.25).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,58.67).
% humidity_at_afternoon(gorizia,57.50).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,42.00).
% humidity_at_afternoon(trieste,55.00).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,54.67).
% humidity_at_afternoon(gemona_stolvizza,32.50).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,63.33).
% humidity_at_afternoon(pordenone,54.17).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"NE",8).
wind_blowing_afternoon(sappada_forni_villa,"NE",14).
wind_blowing_morning(pontebba_tarvisio,"NE",4).
wind_blowing_afternoon(pontebba_tarvisio,"NE",11).
wind_blowing_morning(lignano_grado,"NE",7).
wind_blowing_afternoon(lignano_grado,"NE",12).
wind_blowing_morning(barcis,"NE",8).
wind_blowing_afternoon(barcis,"NE",14).
wind_blowing_morning(udine_palmanova,"E",3).
wind_blowing_afternoon(udine_palmanova,"NE",10).
wind_blowing_morning(gorizia,"E",3).
wind_blowing_afternoon(gorizia,"NE",10).
wind_blowing_morning(trieste,"E",3).
wind_blowing_afternoon(trieste,"NE",10).
wind_blowing_morning(gemona_stolvizza,"NE",4).
wind_blowing_afternoon(gemona_stolvizza,"NE",11).
wind_blowing_morning(pordenone,"NE",7).
wind_blowing_afternoon(pordenone,"NE",12).

temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(barcis,pordenone).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(barcis,sappada_forni_villa).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(lignano_grado,trieste).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(barcis,pordenone).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(barcis,sappada_forni_villa).
temp_front_afternoon_at_1_4km(lignano_grado,pordenone).
temp_front_afternoon_at_1_4km(pordenone,udine_palmanova).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_9km(lignano_grado,pordenone).
temp_front_afternoon_at_9km(pordenone,udine_palmanova).
temp_front_afternoon_at_9km(pontebba_tarvisio,sappada_forni_villa).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(barcis,sappada_forni_villa).
hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(lignano_grado,trieste).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(barcis,sappada_forni_villa).
hum_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gorizia,lignano_grado).
hum_front_morning_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(lignano_grado,trieste).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(lignano_grado,pordenone).
hum_front_afternoon_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_3km(lignano_grado,trieste).
hum_front_afternoon_at_3km(gorizia,trieste).
hum_front_afternoon_at_3km(gemona_stolvizza,gorizia).
hum_front_morning_at_5_5km(gorizia,pontebba_tarvisio).
hum_front_morning_at_9km(lignano_grado,trieste).
hum_front_afternoon_at_9km(sappada_forni_villa,udine_palmanova).

}). 
