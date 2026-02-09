location_considered(gemona_stolvizza). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 10, 24).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_5_5km_covers(pontebba_tarvisio,1926, 0).
cloud_at_5_5km_covers(gemona_stolvizza,1926, 0).
cloud_at_9km_covers(barcis,2321, 18).
cloud_at_9km_covers(pordenone,2321, 18).
cloud_at_9km_covers(sappada_forni_villa,2322, 19).
cloud_at_9km_covers(pontebba_tarvisio,2322, 19).
cloud_at_9km_covers(gemona_stolvizza,2322, 19).
cloud_at_9km_covers(barcis,2321, 19).
cloud_at_9km_covers(lignano_grado,2328, 21).
cloud_at_9km_covers(udine_palmanova,2328, 21).
cloud_at_9km_covers(gorizia,2328, 21).
cloud_at_9km_covers(barcis,2321, 21).
cloud_at_9km_covers(lignano_grado,2329, 22).
cloud_at_9km_covers(trieste,2329, 22).
cloud_at_9km_covers(pordenone,2329, 22).
cloud_at_9km_covers(gorizia,2336, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,275.20).
% temperature_at_afternoon(sappada_forni_villa,275.46).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,277.87).
% temperature_at_afternoon(pontebba_tarvisio,278.42).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.17).
% temperature_at_afternoon(lignano_grado,274.67).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,275.37).
% temperature_at_afternoon(barcis,275.08).
temperature_decreased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.83).
% temperature_at_afternoon(udine_palmanova,275.25).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,276.90).
% temperature_at_afternoon(gorizia,275.81).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,274.73).
% temperature_at_afternoon(trieste,274.83).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.17).
% temperature_at_afternoon(gemona_stolvizza,276.81).
temperature_decreased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.03).
% temperature_at_afternoon(pordenone,275.08).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,61.33).
% humidity_at_afternoon(sappada_forni_villa,59.58).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,49.33).
% humidity_at_afternoon(pontebba_tarvisio,35.42).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,38.00).
% humidity_at_afternoon(lignano_grado,53.75).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,49.33).
% humidity_at_afternoon(barcis,47.08).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,43.33).
% humidity_at_afternoon(udine_palmanova,42.50).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,40.00).
% humidity_at_afternoon(gorizia,42.92).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,50.67).
% humidity_at_afternoon(trieste,47.50).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,59.33).
% humidity_at_afternoon(gemona_stolvizza,33.33).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,48.00).
% humidity_at_afternoon(pordenone,42.92).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",38).
wind_blowing_afternoon(sappada_forni_villa,"E",46).
wind_blowing_morning(pontebba_tarvisio,"E",38).
wind_blowing_afternoon(pontebba_tarvisio,"E",45).
wind_blowing_morning(lignano_grado,"E",41).
wind_blowing_afternoon(lignano_grado,"E",48).
wind_blowing_morning(barcis,"E",38).
wind_blowing_afternoon(barcis,"E",46).
wind_blowing_morning(udine_palmanova,"E",41).
wind_blowing_afternoon(udine_palmanova,"E",47).
wind_blowing_morning(gorizia,"E",41).
wind_blowing_afternoon(gorizia,"E",47).
wind_blowing_morning(trieste,"E",41).
wind_blowing_afternoon(trieste,"E",47).
wind_blowing_morning(gemona_stolvizza,"E",38).
wind_blowing_afternoon(gemona_stolvizza,"E",45).
wind_blowing_morning(pordenone,"E",41).
wind_blowing_afternoon(pordenone,"E",48).

temp_front_morning_at_100m(lignano_grado,udine_palmanova).
temp_front_morning_at_100m(barcis,pordenone).
temp_front_morning_at_100m(gorizia,lignano_grado).
temp_front_morning_at_100m(pordenone,udine_palmanova).
temp_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(pordenone,sappada_forni_villa).
temp_front_afternoon_at_9km(pordenone,udine_palmanova).
temp_front_afternoon_at_9km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_9km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_9km(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_9km(lignano_grado,trieste).

hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_100m(barcis,sappada_forni_villa).
hum_front_afternoon_at_100m(gemona_stolvizza,sappada_forni_villa).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_750m(barcis,sappada_forni_villa).
hum_front_afternoon_at_750m(gemona_stolvizza,sappada_forni_villa).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_1_4km(barcis,sappada_forni_villa).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(gemona_stolvizza,sappada_forni_villa).
hum_front_afternoon_at_3km(lignano_grado,trieste).
hum_front_afternoon_at_3km(gorizia,trieste).
hum_front_afternoon_at_5_5km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_5_5km(barcis,sappada_forni_villa).
hum_front_afternoon_at_5_5km(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_5_5km(pordenone,sappada_forni_villa).
hum_front_morning_at_9km(gorizia,pontebba_tarvisio).
