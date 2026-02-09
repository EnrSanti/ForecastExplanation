location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 6, 29).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,276.53).
% temperature_at_afternoon(sappada_forni_villa,277.31).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,275.73).
% temperature_at_afternoon(pontebba_tarvisio,277.44).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,277.30).
% temperature_at_afternoon(lignano_grado,277.96).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,278.30).
% temperature_at_afternoon(barcis,277.44).
temperature_decreased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,277.17).
% temperature_at_afternoon(udine_palmanova,277.00).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,277.33).
% temperature_at_afternoon(gorizia,277.96).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,277.83).
% temperature_at_afternoon(trieste,278.77).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.13).
% temperature_at_afternoon(gemona_stolvizza,279.04).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.47).
% temperature_at_afternoon(pordenone,275.67).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,59.33).
% humidity_at_afternoon(sappada_forni_villa,48.33).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,66.00).
% humidity_at_afternoon(pontebba_tarvisio,42.08).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,45.33).
% humidity_at_afternoon(lignano_grado,60.83).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,43.33).
% humidity_at_afternoon(barcis,43.75).
humidity_increased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,43.33).
% humidity_at_afternoon(udine_palmanova,53.75).
humidity_increased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,51.33).
% humidity_at_afternoon(gorizia,62.08).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,68.00).
% humidity_at_afternoon(trieste,70.00).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,44.00).
% humidity_at_afternoon(gemona_stolvizza,43.75).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,53.33).
% humidity_at_afternoon(pordenone,55.83).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",12).
wind_blowing_afternoon(sappada_forni_villa,"E",17).
wind_blowing_morning(pontebba_tarvisio,"SE",15).
wind_blowing_afternoon(pontebba_tarvisio,"E",18).
wind_blowing_morning(lignano_grado,"SE",11).
wind_blowing_afternoon(lignano_grado,"E",14).
wind_blowing_morning(barcis,"E",12).
wind_blowing_afternoon(barcis,"E",17).
wind_blowing_morning(udine_palmanova,"SE",15).
wind_blowing_afternoon(udine_palmanova,"E",14).
wind_blowing_morning(gorizia,"SE",15).
wind_blowing_afternoon(gorizia,"E",14).
wind_blowing_morning(trieste,"SE",15).
wind_blowing_afternoon(trieste,"E",14).
wind_blowing_morning(gemona_stolvizza,"SE",15).
wind_blowing_afternoon(gemona_stolvizza,"E",18).
wind_blowing_morning(pordenone,"SE",11).
wind_blowing_afternoon(pordenone,"E",14).

temp_front_morning_at_100m(barcis,sappada_forni_villa).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(lignano_grado,trieste).
temp_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_750m(barcis,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_750m(gorizia,udine_palmanova).
temp_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_1_4km(barcis,pordenone).
temp_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
temp_front_morning_at_3km(barcis,pordenone).
temp_front_morning_at_3km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(lignano_grado,udine_palmanova).
temp_front_afternoon_at_3km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(pordenone,udine_palmanova).
temp_front_afternoon_at_3km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_3km(gemona_stolvizza,sappada_forni_villa).
temp_front_afternoon_at_3km(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_5_5km(barcis,sappada_forni_villa).
temp_front_afternoon_at_5_5km(pordenone,udine_palmanova).
temp_front_morning_at_9km(lignano_grado,trieste).
temp_front_morning_at_9km(gorizia,trieste).

hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(gorizia,lignano_grado).
hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(lignano_grado,pordenone).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(pordenone,udine_palmanova).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_3km(barcis,sappada_forni_villa).
hum_front_morning_at_9km(lignano_grado,trieste).
hum_front_morning_at_9km(gorizia,trieste).
