location_considered(sappada_forni_villa). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 6, 25).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(pordenone,888, 12).
cloud_at_1_4km_covers(udine_palmanova,890, 15).
cloud_at_1_4km_covers(sappada_forni_villa,892, 21).
cloud_at_3km_covers(gemona_stolvizza,994, 12).
cloud_at_5_5km_covers(barcis,989, 8).
cloud_at_9km_covers(trieste,1154, 2).
cloud_at_9km_covers(barcis,1158, 10).
cloud_at_9km_covers(barcis,1158, 11).
cloud_at_9km_covers(gemona_stolvizza,1162, 12).
cloud_at_9km_covers(barcis,1161, 13).
cloud_at_9km_covers(gemona_stolvizza,1162, 13).
cloud_at_9km_covers(gemona_stolvizza,1162, 14).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,275.07).
% temperature_at_afternoon(sappada_forni_villa,278.58).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,273.03).
% temperature_at_afternoon(pontebba_tarvisio,278.98).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,278.67).
% temperature_at_afternoon(lignano_grado,278.52).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,277.63).
% temperature_at_afternoon(barcis,278.65).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.90).
% temperature_at_afternoon(udine_palmanova,277.46).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,277.97).
% temperature_at_afternoon(gorizia,278.54).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,278.77).
% temperature_at_afternoon(trieste,278.71).
temperature_decreased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.43).
% temperature_at_afternoon(gemona_stolvizza,278.81).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.90).
% temperature_at_afternoon(pordenone,277.38).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,53.33).
% humidity_at_afternoon(sappada_forni_villa,39.58).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,58.00).
% humidity_at_afternoon(pontebba_tarvisio,36.67).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,52.00).
% humidity_at_afternoon(lignano_grado,55.00).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,42.67).
% humidity_at_afternoon(barcis,35.42).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,52.00).
% humidity_at_afternoon(udine_palmanova,46.67).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,55.33).
% humidity_at_afternoon(gorizia,52.50).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,48.67).
% humidity_at_afternoon(trieste,56.67).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,44.67).
% humidity_at_afternoon(gemona_stolvizza,38.33).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,53.33).
% humidity_at_afternoon(pordenone,46.67).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",13).
wind_blowing_afternoon(sappada_forni_villa,"E",13).
wind_blowing_morning(pontebba_tarvisio,"SE",14).
wind_blowing_afternoon(pontebba_tarvisio,"E",13).
wind_blowing_morning(lignano_grado,"E",13).
wind_blowing_afternoon(lignano_grado,"E",13).
wind_blowing_morning(barcis,"E",13).
wind_blowing_afternoon(barcis,"E",13).
wind_blowing_morning(udine_palmanova,"SE",12).
wind_blowing_afternoon(udine_palmanova,"E",13).
wind_blowing_morning(gorizia,"SE",12).
wind_blowing_afternoon(gorizia,"E",13).
wind_blowing_morning(trieste,"SE",12).
wind_blowing_afternoon(trieste,"E",13).
wind_blowing_morning(gemona_stolvizza,"SE",14).
wind_blowing_afternoon(gemona_stolvizza,"E",13).
wind_blowing_morning(pordenone,"E",13).
wind_blowing_afternoon(pordenone,"E",13).

temp_front_morning_at_100m(barcis,sappada_forni_villa).
temp_front_morning_at_100m(lignano_grado,udine_palmanova).
temp_front_morning_at_100m(gorizia,lignano_grado).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,udine_palmanova).
temp_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_750m(barcis,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(barcis,sappada_forni_villa).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(gemona_stolvizza,sappada_forni_villa).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_5_5km(gemona_stolvizza,sappada_forni_villa).

hum_front_morning_at_3km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_3km(barcis,pordenone).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_morning_at_3km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_3km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_3km(pordenone,udine_palmanova).
hum_front_afternoon_at_3km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_3km(lignano_grado,pordenone).
hum_front_afternoon_at_3km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_5_5km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_5_5km(gorizia,pontebba_tarvisio).
hum_front_morning_at_9km(barcis,sappada_forni_villa).
