% Example generated data for day (2025, 9, 3)

#pos(e483@1000,{ 

forecasted_sky(trieste, "mostly_clear", autumn),
forecasted_rain(trieste, 0, autumn)},
{
partially_sunny_at(trieste,autumn), 
covered_at(trieste,autumn), 
rains_at(trieste,1,autumn), 
rains_at(trieste,2,autumn), 
rains_at(trieste,4,autumn), 
rains_at(trieste,6,autumn)
},
{
location_considered(trieste). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 3).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_5_5km_covers(udine_palmanova,1343, 0).
cloud_at_5_5km_covers(trieste,1340, 0).
cloud_at_5_5km_covers(barcis,1350, 18).
cloud_at_5_5km_covers(pordenone,1350, 18).
cloud_at_5_5km_covers(udine_palmanova,1350, 21).
cloud_at_9km_covers(udine_palmanova,1588, 0).
cloud_at_9km_covers(trieste,1581, 0).
cloud_at_9km_covers(barcis,1594, 5).
cloud_at_9km_covers(lignano_grado,1596, 6).
cloud_at_9km_covers(pordenone,1596, 6).
cloud_at_9km_covers(udine_palmanova,1596, 7).
cloud_at_9km_covers(barcis,1603, 10).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,272.87).
% temperature_at_afternoon(sappada_forni_villa,277.73).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,275.10).
% temperature_at_afternoon(pontebba_tarvisio,277.98).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.97).
% temperature_at_afternoon(lignano_grado,277.75).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,274.27).
% temperature_at_afternoon(barcis,278.92).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.20).
% temperature_at_afternoon(udine_palmanova,277.52).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,276.73).
% temperature_at_afternoon(gorizia,278.58).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,276.10).
% temperature_at_afternoon(trieste,279.25).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,275.33).
% temperature_at_afternoon(gemona_stolvizza,279.06).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.37).
% temperature_at_afternoon(pordenone,277.25).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,60.00).
% humidity_at_afternoon(sappada_forni_villa,34.17).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,41.33).
% humidity_at_afternoon(pontebba_tarvisio,30.83).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,48.00).
% humidity_at_afternoon(lignano_grado,57.08).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,52.67).
% humidity_at_afternoon(barcis,29.17).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,44.67).
% humidity_at_afternoon(udine_palmanova,40.00).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,44.00).
% humidity_at_afternoon(gorizia,41.67).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,40.67).
% humidity_at_afternoon(trieste,51.25).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,40.67).
% humidity_at_afternoon(gemona_stolvizza,35.00).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,46.00).
% humidity_at_afternoon(pordenone,55.42).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"SE",23).
wind_blowing_afternoon(sappada_forni_villa,"E",10).
wind_blowing_morning(pontebba_tarvisio,"SE",26).
wind_blowing_afternoon(pontebba_tarvisio,"SE",14).
wind_blowing_morning(lignano_grado,"SE",25).
wind_blowing_afternoon(lignano_grado,"SE",11).
wind_blowing_morning(barcis,"SE",23).
wind_blowing_afternoon(barcis,"E",10).
wind_blowing_morning(udine_palmanova,"SE",26).
wind_blowing_afternoon(udine_palmanova,"SE",15).
wind_blowing_morning(gorizia,"SE",26).
wind_blowing_afternoon(gorizia,"SE",15).
wind_blowing_morning(trieste,"SE",26).
wind_blowing_afternoon(trieste,"SE",15).
wind_blowing_morning(gemona_stolvizza,"SE",26).
wind_blowing_afternoon(gemona_stolvizza,"SE",14).
wind_blowing_morning(pordenone,"SE",25).
wind_blowing_afternoon(pordenone,"SE",11).

temp_front_morning_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(lignano_grado,trieste).
temp_front_afternoon_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(barcis,sappada_forni_villa).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_morning_at_3km(lignano_grado,udine_palmanova).
temp_front_afternoon_at_3km(lignano_grado,udine_palmanova).
temp_front_afternoon_at_3km(gorizia,lignano_grado).
temp_front_afternoon_at_5_5km(lignano_grado,trieste).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(lignano_grado,pordenone).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(pordenone,udine_palmanova).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gemona_stolvizza,sappada_forni_villa).
hum_front_morning_at_1_4km(pordenone,sappada_forni_villa).
hum_front_morning_at_3km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_3km(lignano_grado,trieste).
hum_front_afternoon_at_3km(gorizia,trieste).
hum_front_morning_at_5_5km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_5_5km(lignano_grado,pordenone).
hum_front_morning_at_5_5km(gemona_stolvizza,sappada_forni_villa).
hum_front_morning_at_5_5km(pordenone,udine_palmanova).
hum_front_afternoon_at_5_5km(lignano_grado,trieste).
hum_front_afternoon_at_9km(pordenone,udine_palmanova).
hum_front_afternoon_at_9km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_9km(gemona_stolvizza,pontebba_tarvisio).

}). 
