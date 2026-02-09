location_considered(barcis). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 6, 30).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(sappada_forni_villa,930, 11).
cloud_at_1_4km_covers(gorizia,932, 11).
cloud_at_1_4km_covers(sappada_forni_villa,930, 12).
cloud_at_1_4km_covers(gemona_stolvizza,932, 12).
cloud_at_3km_covers(barcis,1094, 9).
cloud_at_3km_covers(sappada_forni_villa,1094, 10).
cloud_at_3km_covers(sappada_forni_villa,1103, 11).
cloud_at_3km_covers(sappada_forni_villa,1103, 12).
cloud_at_5_5km_covers(barcis,1049, 2).
cloud_at_5_5km_covers(barcis,1053, 9).
cloud_at_5_5km_covers(sappada_forni_villa,1053, 10).
cloud_at_5_5km_covers(sappada_forni_villa,1053, 11).
cloud_at_5_5km_covers(gorizia,1066, 22).
cloud_at_5_5km_covers(gemona_stolvizza,1066, 22).
cloud_at_9km_covers(pontebba_tarvisio,1285, 6).
cloud_at_9km_covers(pontebba_tarvisio,1285, 7).
cloud_at_9km_covers(sappada_forni_villa,1285, 8).
cloud_at_9km_covers(pontebba_tarvisio,1285, 8).
cloud_at_9km_covers(sappada_forni_villa,1285, 9).
cloud_at_9km_covers(pontebba_tarvisio,1285, 9).
cloud_at_9km_covers(barcis,1285, 9).
cloud_at_9km_covers(gemona_stolvizza,1285, 9).
cloud_at_9km_covers(sappada_forni_villa,1288, 10).
cloud_at_9km_covers(pontebba_tarvisio,1288, 10).
cloud_at_9km_covers(gemona_stolvizza,1288, 10).
cloud_at_9km_covers(sappada_forni_villa,1288, 11).
cloud_at_9km_covers(pontebba_tarvisio,1288, 11).
cloud_at_9km_covers(gemona_stolvizza,1288, 11).
cloud_at_9km_covers(sappada_forni_villa,1293, 12).
cloud_at_9km_covers(pontebba_tarvisio,1293, 12).
cloud_at_9km_covers(sappada_forni_villa,1300, 16).
cloud_at_9km_covers(pontebba_tarvisio,1300, 16).
cloud_at_9km_covers(barcis,1300, 16).
cloud_at_9km_covers(sappada_forni_villa,1300, 17).
cloud_at_9km_covers(pontebba_tarvisio,1300, 17).
cloud_at_9km_covers(barcis,1300, 17).
cloud_at_9km_covers(gemona_stolvizza,1300, 17).
cloud_at_9km_covers(pordenone,1300, 17).
cloud_at_9km_covers(sappada_forni_villa,1300, 18).
cloud_at_9km_covers(pontebba_tarvisio,1300, 18).
cloud_at_9km_covers(barcis,1300, 18).
cloud_at_9km_covers(udine_palmanova,1300, 18).
cloud_at_9km_covers(gorizia,1300, 18).
cloud_at_9km_covers(gemona_stolvizza,1300, 18).
cloud_at_9km_covers(sappada_forni_villa,1300, 19).
cloud_at_9km_covers(pontebba_tarvisio,1300, 19).
cloud_at_9km_covers(barcis,1300, 19).
cloud_at_9km_covers(udine_palmanova,1300, 19).
cloud_at_9km_covers(gorizia,1300, 19).
cloud_at_9km_covers(gemona_stolvizza,1300, 19).
cloud_at_9km_covers(pordenone,1300, 19).
cloud_at_9km_covers(sappada_forni_villa,1300, 20).
cloud_at_9km_covers(pontebba_tarvisio,1300, 20).
cloud_at_9km_covers(barcis,1300, 20).
cloud_at_9km_covers(udine_palmanova,1300, 20).
cloud_at_9km_covers(gorizia,1300, 20).
cloud_at_9km_covers(gemona_stolvizza,1300, 20).
cloud_at_9km_covers(pontebba_tarvisio,1300, 21).
cloud_at_9km_covers(lignano_grado,1300, 21).
cloud_at_9km_covers(barcis,1300, 21).
cloud_at_9km_covers(udine_palmanova,1300, 21).
cloud_at_9km_covers(gorizia,1300, 21).
cloud_at_9km_covers(trieste,1300, 21).
cloud_at_9km_covers(pordenone,1300, 21).
cloud_at_9km_covers(sappada_forni_villa,1300, 22).
cloud_at_9km_covers(pontebba_tarvisio,1300, 22).
cloud_at_9km_covers(lignano_grado,1300, 22).
cloud_at_9km_covers(barcis,1300, 22).
cloud_at_9km_covers(pordenone,1300, 22).
cloud_at_9km_covers(sappada_forni_villa,1300, 23).
cloud_at_9km_covers(pontebba_tarvisio,1300, 23).
cloud_at_9km_covers(lignano_grado,1300, 23).
cloud_at_9km_covers(barcis,1300, 23).
cloud_at_9km_covers(udine_palmanova,1300, 23).
cloud_at_9km_covers(gorizia,1300, 23).
cloud_at_9km_covers(gemona_stolvizza,1300, 23).
cloud_at_9km_covers(pordenone,1300, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,276.63).
% temperature_at_afternoon(sappada_forni_villa,278.69).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,276.03).
% temperature_at_afternoon(pontebba_tarvisio,279.25).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.87).
% temperature_at_afternoon(lignano_grado,276.29).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,277.33).
% temperature_at_afternoon(barcis,276.96).
temperature_decreased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.63).
% temperature_at_afternoon(udine_palmanova,276.15).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,277.67).
% temperature_at_afternoon(gorizia,277.33).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,276.80).
% temperature_at_afternoon(trieste,278.00).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.53).
% temperature_at_afternoon(gemona_stolvizza,278.71).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,274.93).
% temperature_at_afternoon(pordenone,274.77).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,24.00).
% humidity_at_afternoon(sappada_forni_villa,23.33).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,31.33).
% humidity_at_afternoon(pontebba_tarvisio,24.58).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,62.67).
% humidity_at_afternoon(lignano_grado,47.50).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,23.33).
% humidity_at_afternoon(barcis,40.00).
humidity_increased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,50.67).
% humidity_at_afternoon(udine_palmanova,47.92).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,50.00).
% humidity_at_afternoon(gorizia,57.08).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,64.67).
% humidity_at_afternoon(trieste,58.33).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,34.00).
% humidity_at_afternoon(gemona_stolvizza,33.33).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,60.00).
% humidity_at_afternoon(pordenone,44.58).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",18).
wind_blowing_afternoon(sappada_forni_villa,"E",21).
wind_blowing_morning(pontebba_tarvisio,"E",17).
wind_blowing_afternoon(pontebba_tarvisio,"E",20).
wind_blowing_morning(lignano_grado,"E",20).
wind_blowing_afternoon(lignano_grado,"E",22).
wind_blowing_morning(barcis,"E",18).
wind_blowing_afternoon(barcis,"E",21).
wind_blowing_morning(udine_palmanova,"E",20).
wind_blowing_afternoon(udine_palmanova,"E",21).
wind_blowing_morning(gorizia,"E",20).
wind_blowing_afternoon(gorizia,"E",21).
wind_blowing_morning(trieste,"E",20).
wind_blowing_afternoon(trieste,"E",21).
wind_blowing_morning(gemona_stolvizza,"E",17).
wind_blowing_afternoon(gemona_stolvizza,"E",20).
wind_blowing_morning(pordenone,"E",20).
wind_blowing_afternoon(pordenone,"E",22).

temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,sappada_forni_villa).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(lignano_grado,pordenone).
temp_front_afternoon_at_750m(barcis,sappada_forni_villa).
temp_front_afternoon_at_750m(pordenone,udine_palmanova).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(lignano_grado,trieste).
temp_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(barcis,pordenone).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_5_5km(barcis,pordenone).

hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(pordenone,udine_palmanova).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(pordenone,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(lignano_grado,pordenone).
hum_front_morning_at_1_4km(barcis,pordenone).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_morning_at_3km(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_3km(lignano_grado,trieste).
hum_front_afternoon_at_3km(gorizia,lignano_grado).
hum_front_morning_at_9km(lignano_grado,udine_palmanova).
hum_front_morning_at_9km(gorizia,lignano_grado).
hum_front_afternoon_at_9km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_9km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_9km(pordenone,sappada_forni_villa).
