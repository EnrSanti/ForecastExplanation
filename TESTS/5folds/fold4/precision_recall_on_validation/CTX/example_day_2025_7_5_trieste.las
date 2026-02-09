location_considered(trieste). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 7, 5).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(sappada_forni_villa,1124, 10).
cloud_at_5_5km_covers(sappada_forni_villa,1072, 10).
cloud_at_5_5km_covers(pordenone,1088, 15).
cloud_at_9km_covers(lignano_grado,1310, 1).
cloud_at_9km_covers(pordenone,1310, 1).
cloud_at_9km_covers(lignano_grado,1309, 2).
cloud_at_9km_covers(gorizia,1309, 2).
cloud_at_9km_covers(gorizia,1300, 3).
cloud_at_9km_covers(sappada_forni_villa,1324, 10).
cloud_at_9km_covers(barcis,1324, 10).
cloud_at_9km_covers(pontebba_tarvisio,1320, 10).
cloud_at_9km_covers(trieste,1322, 10).
cloud_at_9km_covers(sappada_forni_villa,1329, 11).
cloud_at_9km_covers(pontebba_tarvisio,1329, 11).
cloud_at_9km_covers(barcis,1329, 11).
cloud_at_9km_covers(trieste,1322, 11).
cloud_at_9km_covers(sappada_forni_villa,1328, 12).
cloud_at_9km_covers(pontebba_tarvisio,1328, 12).
cloud_at_9km_covers(barcis,1328, 12).
cloud_at_9km_covers(gemona_stolvizza,1328, 12).
cloud_at_9km_covers(sappada_forni_villa,1332, 13).
cloud_at_9km_covers(pontebba_tarvisio,1332, 13).
cloud_at_9km_covers(barcis,1332, 13).
cloud_at_9km_covers(gemona_stolvizza,1332, 13).
cloud_at_9km_covers(trieste,1333, 13).
cloud_at_9km_covers(sappada_forni_villa,1332, 14).
cloud_at_9km_covers(pontebba_tarvisio,1332, 14).
cloud_at_9km_covers(barcis,1332, 14).
cloud_at_9km_covers(gemona_stolvizza,1332, 14).
cloud_at_9km_covers(pontebba_tarvisio,1332, 15).
cloud_at_9km_covers(gemona_stolvizza,1332, 15).
cloud_at_9km_covers(gemona_stolvizza,1332, 16).
cloud_at_9km_covers(sappada_forni_villa,1332, 20).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,276.90).
% temperature_at_afternoon(sappada_forni_villa,278.31).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.20).
% temperature_at_afternoon(pontebba_tarvisio,277.79).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,275.13).
% temperature_at_afternoon(lignano_grado,276.60).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,277.43).
% temperature_at_afternoon(barcis,278.25).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.67).
% temperature_at_afternoon(udine_palmanova,275.85).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,276.10).
% temperature_at_afternoon(gorizia,276.19).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,275.20).
% temperature_at_afternoon(trieste,275.85).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.07).
% temperature_at_afternoon(gemona_stolvizza,277.92).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.20).
% temperature_at_afternoon(pordenone,276.88).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,38.00).
% humidity_at_afternoon(sappada_forni_villa,54.17).
humidity_increased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,24.00).
% humidity_at_afternoon(pontebba_tarvisio,48.33).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,44.00).
% humidity_at_afternoon(lignano_grado,54.58).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,37.33).
% humidity_at_afternoon(barcis,51.67).
humidity_increased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,56.00).
% humidity_at_afternoon(udine_palmanova,46.67).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,69.33).
% humidity_at_afternoon(gorizia,61.67).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,70.00).
% humidity_at_afternoon(trieste,67.92).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,52.00).
% humidity_at_afternoon(gemona_stolvizza,53.75).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,49.33).
% humidity_at_afternoon(pordenone,57.08).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",15).
wind_blowing_afternoon(sappada_forni_villa,"NE",12).
wind_blowing_morning(pontebba_tarvisio,"E",16).
wind_blowing_afternoon(pontebba_tarvisio,"E",12).
wind_blowing_morning(lignano_grado,"E",17).
wind_blowing_afternoon(lignano_grado,"E",15).
wind_blowing_morning(barcis,"E",15).
wind_blowing_afternoon(barcis,"NE",12).
wind_blowing_morning(udine_palmanova,"E",16).
wind_blowing_afternoon(udine_palmanova,"E",14).
wind_blowing_morning(gorizia,"E",16).
wind_blowing_afternoon(gorizia,"E",14).
wind_blowing_morning(trieste,"E",16).
wind_blowing_afternoon(trieste,"E",14).
wind_blowing_morning(gemona_stolvizza,"E",16).
wind_blowing_afternoon(gemona_stolvizza,"E",12).
wind_blowing_morning(pordenone,"E",17).
wind_blowing_afternoon(pordenone,"E",15).

temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,lignano_grado).
temp_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_750m(barcis,sappada_forni_villa).
temp_front_morning_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(lignano_grado,pordenone).
temp_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_750m(pordenone,udine_palmanova).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(barcis,pordenone).
temp_front_morning_at_1_4km(lignano_grado,udine_palmanova).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,gorizia).
temp_front_afternoon_at_1_4km(lignano_grado,trieste).
temp_front_afternoon_at_1_4km(gorizia,trieste).
temp_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
temp_front_afternoon_at_1_4km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(pordenone,udine_palmanova).
temp_front_afternoon_at_3km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(lignano_grado,trieste).
temp_front_morning_at_9km(gorizia,pontebba_tarvisio).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_750m(barcis,pordenone).
hum_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_750m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(gemona_stolvizza,gorizia).
hum_front_morning_at_1_4km(lignano_grado,udine_palmanova).
hum_front_morning_at_1_4km(barcis,pordenone).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,gorizia).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_morning_at_3km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_3km(pordenone,udine_palmanova).
hum_front_morning_at_5_5km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_5_5km(pordenone,sappada_forni_villa).
