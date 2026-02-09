location_considered(pontebba_tarvisio). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 6, 28).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(sappada_forni_villa,1060, 0).
cloud_at_3km_covers(pontebba_tarvisio,1056, 0).
cloud_at_3km_covers(sappada_forni_villa,1063, 2).
cloud_at_3km_covers(barcis,1071, 10).
cloud_at_3km_covers(sappada_forni_villa,1072, 11).
cloud_at_3km_covers(pontebba_tarvisio,1072, 11).
cloud_at_3km_covers(sappada_forni_villa,1076, 12).
cloud_at_3km_covers(pontebba_tarvisio,1076, 12).
cloud_at_3km_covers(barcis,1076, 12).
cloud_at_3km_covers(pordenone,1080, 13).
cloud_at_3km_covers(pontebba_tarvisio,1081, 14).
cloud_at_3km_covers(pontebba_tarvisio,1083, 15).
cloud_at_3km_covers(pontebba_tarvisio,1078, 17).
cloud_at_9km_covers(pontebba_tarvisio,1266, 2).
cloud_at_9km_covers(udine_palmanova,1266, 2).
cloud_at_9km_covers(gemona_stolvizza,1266, 2).
cloud_at_9km_covers(sappada_forni_villa,1269, 4).
cloud_at_9km_covers(barcis,1269, 4).
cloud_at_9km_covers(gemona_stolvizza,1269, 4).
cloud_at_9km_covers(pordenone,1269, 4).
cloud_at_9km_covers(gorizia,1270, 4).
cloud_at_9km_covers(trieste,1270, 4).
cloud_at_9km_covers(lignano_grado,1269, 5).
cloud_at_9km_covers(sappada_forni_villa,1273, 11).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,272.93).
% temperature_at_afternoon(sappada_forni_villa,275.67).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,274.83).
% temperature_at_afternoon(pontebba_tarvisio,273.06).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,274.60).
% temperature_at_afternoon(lignano_grado,277.65).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,274.23).
% temperature_at_afternoon(barcis,275.44).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,274.23).
% temperature_at_afternoon(udine_palmanova,276.44).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,274.87).
% temperature_at_afternoon(gorizia,278.10).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,276.00).
% temperature_at_afternoon(trieste,278.50).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,275.23).
% temperature_at_afternoon(gemona_stolvizza,277.69).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,273.47).
% temperature_at_afternoon(pordenone,274.67).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,66.67).
% humidity_at_afternoon(sappada_forni_villa,43.75).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,56.00).
% humidity_at_afternoon(pontebba_tarvisio,54.58).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,68.00).
% humidity_at_afternoon(lignano_grado,62.08).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,64.67).
% humidity_at_afternoon(barcis,53.33).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,64.00).
% humidity_at_afternoon(udine_palmanova,50.00).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,68.67).
% humidity_at_afternoon(gorizia,61.67).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,75.33).
% humidity_at_afternoon(trieste,62.08).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,51.33).
% humidity_at_afternoon(gemona_stolvizza,44.58).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,54.67).
% humidity_at_afternoon(pordenone,48.75).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",23).
wind_blowing_afternoon(sappada_forni_villa,"S",21).
wind_blowing_morning(pontebba_tarvisio,"S",24).
wind_blowing_afternoon(pontebba_tarvisio,"S",22).
wind_blowing_morning(lignano_grado,"S",23).
wind_blowing_afternoon(lignano_grado,"S",21).
wind_blowing_morning(barcis,"S",23).
wind_blowing_afternoon(barcis,"S",21).
wind_blowing_morning(udine_palmanova,"S",24).
wind_blowing_afternoon(udine_palmanova,"S",22).
wind_blowing_morning(gorizia,"S",24).
wind_blowing_afternoon(gorizia,"S",22).
wind_blowing_morning(trieste,"S",24).
wind_blowing_afternoon(trieste,"S",22).
wind_blowing_morning(gemona_stolvizza,"S",24).
wind_blowing_afternoon(gemona_stolvizza,"S",22).
wind_blowing_morning(pordenone,"S",23).
wind_blowing_afternoon(pordenone,"S",21).

temp_front_afternoon_at_750m(gemona_stolvizza,gorizia).
temp_front_afternoon_at_750m(lignano_grado,trieste).
temp_front_afternoon_at_750m(gorizia,lignano_grado).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_1_4km(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_1_4km(pordenone,udine_palmanova).
temp_front_afternoon_at_1_4km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(lignano_grado,udine_palmanova).
temp_front_afternoon_at_3km(pordenone,udine_palmanova).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_3km(barcis,sappada_forni_villa).
hum_front_morning_at_3km(gemona_stolvizza,gorizia).
