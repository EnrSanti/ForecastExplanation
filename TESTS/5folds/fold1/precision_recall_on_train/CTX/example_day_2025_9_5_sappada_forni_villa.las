location_considered(sappada_forni_villa). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 5).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(sappada_forni_villa,1413, 17).
cloud_at_3km_covers(gorizia,1432, 19).
cloud_at_5_5km_covers(barcis,1365, 13).
cloud_at_5_5km_covers(gorizia,1382, 19).
cloud_at_9km_covers(barcis,1635, 11).
cloud_at_9km_covers(sappada_forni_villa,1635, 12).
cloud_at_9km_covers(barcis,1635, 12).
cloud_at_9km_covers(barcis,1638, 13).
cloud_at_9km_covers(udine_palmanova,1638, 13).
cloud_at_9km_covers(pordenone,1638, 13).
cloud_at_9km_covers(sappada_forni_villa,1642, 14).
cloud_at_9km_covers(pontebba_tarvisio,1640, 17).
cloud_at_9km_covers(gorizia,1658, 18).
cloud_at_9km_covers(gorizia,1660, 19).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,273.37).
% temperature_at_afternoon(sappada_forni_villa,276.60).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,273.37).
% temperature_at_afternoon(pontebba_tarvisio,274.60).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,273.07).
% temperature_at_afternoon(lignano_grado,272.25).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,274.70).
% temperature_at_afternoon(barcis,275.71).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,273.67).
% temperature_at_afternoon(udine_palmanova,272.56).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,272.77).
% temperature_at_afternoon(gorizia,272.19).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,273.03).
% temperature_at_afternoon(trieste,272.54).
temperature_decreased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,273.77).
% temperature_at_afternoon(gemona_stolvizza,274.52).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,274.43).
% temperature_at_afternoon(pordenone,273.21).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,43.33).
% humidity_at_afternoon(sappada_forni_villa,51.25).
humidity_increased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,49.33).
% humidity_at_afternoon(pontebba_tarvisio,52.50).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,61.33).
% humidity_at_afternoon(lignano_grado,57.92).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,46.67).
% humidity_at_afternoon(barcis,45.42).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,64.00).
% humidity_at_afternoon(udine_palmanova,60.00).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,67.33).
% humidity_at_afternoon(gorizia,58.33).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,68.67).
% humidity_at_afternoon(trieste,57.08).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,63.33).
% humidity_at_afternoon(gemona_stolvizza,47.50).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,46.67).
% humidity_at_afternoon(pordenone,51.67).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"NE",19).
wind_blowing_afternoon(sappada_forni_villa,"NE",22).
wind_blowing_morning(pontebba_tarvisio,"NE",18).
wind_blowing_afternoon(pontebba_tarvisio,"NE",18).
wind_blowing_morning(lignano_grado,"NE",18).
wind_blowing_afternoon(lignano_grado,"NE",17).
wind_blowing_morning(barcis,"NE",19).
wind_blowing_afternoon(barcis,"NE",22).
wind_blowing_morning(udine_palmanova,"NE",16).
wind_blowing_afternoon(udine_palmanova,"NE",15).
wind_blowing_morning(gorizia,"NE",16).
wind_blowing_afternoon(gorizia,"NE",15).
wind_blowing_morning(trieste,"NE",16).
wind_blowing_afternoon(trieste,"NE",15).
wind_blowing_morning(gemona_stolvizza,"NE",18).
wind_blowing_afternoon(gemona_stolvizza,"NE",18).
wind_blowing_morning(pordenone,"NE",18).
wind_blowing_afternoon(pordenone,"NE",17).

temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_9km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_9km(gorizia,pontebba_tarvisio).

hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,udine_palmanova).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_morning_at_750m(gorizia,lignano_grado).
hum_front_morning_at_750m(pordenone,udine_palmanova).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(barcis,sappada_forni_villa).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(lignano_grado,trieste).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_9km(pordenone,udine_palmanova).
