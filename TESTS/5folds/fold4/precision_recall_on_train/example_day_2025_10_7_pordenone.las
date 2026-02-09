% Example generated data for day (2025, 10, 7)

#pos(e569@1000,{ 

forecasted_sky(pordenone, "mostly_clear", autumn),
forecasted_rain(pordenone, 0, autumn)},
{
partially_sunny_at(pordenone,autumn), 
covered_at(pordenone,autumn), 
rains_at(pordenone,1,autumn), 
rains_at(pordenone,2,autumn), 
rains_at(pordenone,4,autumn), 
rains_at(pordenone,6,autumn)
},
{
location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 10, 7).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(trieste,1743, 8).
cloud_at_5_5km_covers(gemona_stolvizza,1689, 10).
cloud_at_5_5km_covers(pordenone,1690, 10).
cloud_at_9km_covers(pontebba_tarvisio,1991, 5).
cloud_at_9km_covers(pontebba_tarvisio,1978, 6).
cloud_at_9km_covers(gemona_stolvizza,1978, 6).
cloud_at_9km_covers(barcis,1993, 6).
cloud_at_9km_covers(pontebba_tarvisio,2002, 19).
cloud_at_9km_covers(barcis,2002, 19).
cloud_at_9km_covers(sappada_forni_villa,2002, 20).
cloud_at_9km_covers(pordenone,2002, 20).
cloud_at_9km_covers(pontebba_tarvisio,2005, 21).
cloud_at_9km_covers(lignano_grado,2002, 21).
cloud_at_9km_covers(udine_palmanova,2002, 21).
cloud_at_9km_covers(gemona_stolvizza,2002, 21).
cloud_at_9km_covers(trieste,2009, 22).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,275.47).
% temperature_at_afternoon(sappada_forni_villa,275.38).
temperature_decreased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.53).
% temperature_at_afternoon(pontebba_tarvisio,278.40).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.63).
% temperature_at_afternoon(lignano_grado,275.42).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,272.43).
% temperature_at_afternoon(barcis,274.19).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.07).
% temperature_at_afternoon(udine_palmanova,275.60).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,275.77).
% temperature_at_afternoon(gorizia,275.67).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,274.50).
% temperature_at_afternoon(trieste,275.90).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,275.60).
% temperature_at_afternoon(gemona_stolvizza,276.17).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,275.00).
% temperature_at_afternoon(pordenone,274.12).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,66.00).
% humidity_at_afternoon(sappada_forni_villa,55.83).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,30.67).
% humidity_at_afternoon(pontebba_tarvisio,49.17).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,42.00).
% humidity_at_afternoon(lignano_grado,51.25).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,65.33).
% humidity_at_afternoon(barcis,57.50).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,58.00).
% humidity_at_afternoon(udine_palmanova,51.67).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,69.33).
% humidity_at_afternoon(gorizia,62.92).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,64.67).
% humidity_at_afternoon(trieste,65.00).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,74.67).
% humidity_at_afternoon(gemona_stolvizza,56.67).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,34.67).
% humidity_at_afternoon(pordenone,39.17).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",46).
wind_blowing_afternoon(sappada_forni_villa,"S",38).
wind_blowing_morning(pontebba_tarvisio,"S",50).
wind_blowing_afternoon(pontebba_tarvisio,"S",42).
wind_blowing_morning(lignano_grado,"S",48).
wind_blowing_afternoon(lignano_grado,"S",40).
wind_blowing_morning(barcis,"S",46).
wind_blowing_afternoon(barcis,"S",38).
wind_blowing_morning(udine_palmanova,"S",50).
wind_blowing_afternoon(udine_palmanova,"S",42).
wind_blowing_morning(gorizia,"S",50).
wind_blowing_afternoon(gorizia,"S",42).
wind_blowing_morning(trieste,"S",50).
wind_blowing_afternoon(trieste,"S",42).
wind_blowing_morning(gemona_stolvizza,"S",50).
wind_blowing_afternoon(gemona_stolvizza,"S",42).
wind_blowing_morning(pordenone,"S",48).
wind_blowing_afternoon(pordenone,"S",40).

temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_100m(barcis,pordenone).
temp_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_3km(gorizia,pontebba_tarvisio).
temp_front_morning_at_3km(gemona_stolvizza,gorizia).
temp_front_morning_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_5_5km(lignano_grado,pordenone).
temp_front_morning_at_5_5km(gemona_stolvizza,sappada_forni_villa).
temp_front_morning_at_5_5km(pordenone,udine_palmanova).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_5_5km(barcis,pordenone).
temp_front_afternoon_at_5_5km(gemona_stolvizza,sappada_forni_villa).
temp_front_afternoon_at_5_5km(pordenone,sappada_forni_villa).
temp_front_morning_at_9km(pontebba_tarvisio,sappada_forni_villa).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_750m(barcis,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_5_5km(lignano_grado,trieste).
hum_front_afternoon_at_9km(barcis,sappada_forni_villa).
hum_front_afternoon_at_9km(pordenone,udine_palmanova).

}). 
