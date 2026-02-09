location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 8).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(sappada_forni_villa,1173, 17).
cloud_at_1_4km_covers(sappada_forni_villa,1173, 18).
cloud_at_5_5km_covers(pontebba_tarvisio,1408, 22).
cloud_at_5_5km_covers(lignano_grado,1401, 23).
cloud_at_9km_covers(sappada_forni_villa,1723, 8).
cloud_at_9km_covers(pontebba_tarvisio,1723, 8).
cloud_at_9km_covers(barcis,1723, 8).
cloud_at_9km_covers(gemona_stolvizza,1723, 8).
cloud_at_9km_covers(gorizia,1723, 9).
cloud_at_9km_covers(barcis,1728, 10).
cloud_at_9km_covers(pordenone,1728, 10).
cloud_at_9km_covers(pontebba_tarvisio,1728, 11).
cloud_at_9km_covers(lignano_grado,1733, 11).
cloud_at_9km_covers(udine_palmanova,1733, 11).
cloud_at_9km_covers(sappada_forni_villa,1735, 12).
cloud_at_9km_covers(pontebba_tarvisio,1735, 12).
cloud_at_9km_covers(gemona_stolvizza,1735, 12).
cloud_at_9km_covers(barcis,1736, 12).
cloud_at_9km_covers(gorizia,1733, 12).
cloud_at_9km_covers(trieste,1733, 12).
cloud_at_9km_covers(sappada_forni_villa,1736, 13).
cloud_at_9km_covers(pontebba_tarvisio,1736, 13).
cloud_at_9km_covers(udine_palmanova,1736, 13).
cloud_at_9km_covers(gorizia,1736, 13).
cloud_at_9km_covers(gemona_stolvizza,1736, 13).
cloud_at_9km_covers(sappada_forni_villa,1736, 14).
cloud_at_9km_covers(pontebba_tarvisio,1736, 14).
cloud_at_9km_covers(gorizia,1736, 14).
cloud_at_9km_covers(gemona_stolvizza,1736, 14).
cloud_at_9km_covers(sappada_forni_villa,1736, 15).
cloud_at_9km_covers(pontebba_tarvisio,1736, 15).
cloud_at_9km_covers(barcis,1736, 15).
cloud_at_9km_covers(gemona_stolvizza,1736, 15).
cloud_at_9km_covers(sappada_forni_villa,1738, 16).
cloud_at_9km_covers(pontebba_tarvisio,1738, 16).
cloud_at_9km_covers(sappada_forni_villa,1739, 17).
cloud_at_9km_covers(pontebba_tarvisio,1739, 17).
cloud_at_9km_covers(pontebba_tarvisio,1738, 18).
cloud_at_9km_covers(lignano_grado,1738, 18).
cloud_at_9km_covers(barcis,1738, 18).
cloud_at_9km_covers(trieste,1738, 18).
cloud_at_9km_covers(pordenone,1738, 18).
cloud_at_9km_covers(gemona_stolvizza,1739, 18).
cloud_at_9km_covers(sappada_forni_villa,1739, 19).
cloud_at_9km_covers(pontebba_tarvisio,1739, 19).
cloud_at_9km_covers(lignano_grado,1739, 19).
cloud_at_9km_covers(udine_palmanova,1739, 19).
cloud_at_9km_covers(gorizia,1739, 19).
cloud_at_9km_covers(trieste,1739, 19).
cloud_at_9km_covers(gemona_stolvizza,1739, 19).
cloud_at_9km_covers(pordenone,1739, 19).
cloud_at_9km_covers(pontebba_tarvisio,1739, 20).
cloud_at_9km_covers(trieste,1739, 20).
cloud_at_9km_covers(gemona_stolvizza,1739, 20).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,273.67).
% temperature_at_afternoon(sappada_forni_villa,278.92).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,274.63).
% temperature_at_afternoon(pontebba_tarvisio,277.46).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,278.30).
% temperature_at_afternoon(lignano_grado,278.33).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,276.23).
% temperature_at_afternoon(barcis,278.27).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,277.93).
% temperature_at_afternoon(udine_palmanova,277.65).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,277.97).
% temperature_at_afternoon(gorizia,278.17).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,277.67).
% temperature_at_afternoon(trieste,278.10).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.17).
% temperature_at_afternoon(gemona_stolvizza,278.46).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,277.80).
% temperature_at_afternoon(pordenone,277.81).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,59.33).
% humidity_at_afternoon(sappada_forni_villa,38.75).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,59.33).
% humidity_at_afternoon(pontebba_tarvisio,42.08).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,60.67).
% humidity_at_afternoon(lignano_grado,53.75).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,47.33).
% humidity_at_afternoon(barcis,26.67).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,52.00).
% humidity_at_afternoon(udine_palmanova,47.92).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,64.00).
% humidity_at_afternoon(gorizia,50.00).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,73.33).
% humidity_at_afternoon(trieste,53.75).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,46.67).
% humidity_at_afternoon(gemona_stolvizza,50.42).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,54.67).
% humidity_at_afternoon(pordenone,47.50).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"SE",19).
wind_blowing_afternoon(sappada_forni_villa,"E",17).
wind_blowing_morning(pontebba_tarvisio,"SE",22).
wind_blowing_afternoon(pontebba_tarvisio,"E",18).
wind_blowing_morning(lignano_grado,"SE",17).
wind_blowing_afternoon(lignano_grado,"E",17).
wind_blowing_morning(barcis,"SE",19).
wind_blowing_afternoon(barcis,"E",17).
wind_blowing_morning(udine_palmanova,"SE",18).
wind_blowing_afternoon(udine_palmanova,"E",18).
wind_blowing_morning(gorizia,"SE",18).
wind_blowing_afternoon(gorizia,"E",18).
wind_blowing_morning(trieste,"SE",18).
wind_blowing_afternoon(trieste,"E",18).
wind_blowing_morning(gemona_stolvizza,"SE",22).
wind_blowing_afternoon(gemona_stolvizza,"E",18).
wind_blowing_morning(pordenone,"SE",17).
wind_blowing_afternoon(pordenone,"E",17).

temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(barcis,pordenone).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,udine_palmanova).
temp_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_750m(barcis,pordenone).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_750m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(gorizia,trieste).
temp_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_5_5km(barcis,sappada_forni_villa).
temp_front_morning_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_5_5km(pordenone,sappada_forni_villa).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(barcis,sappada_forni_villa).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(lignano_grado,trieste).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_1_4km(barcis,sappada_forni_villa).
hum_front_afternoon_at_1_4km(lignano_grado,trieste).
hum_front_afternoon_at_1_4km(gorizia,trieste).
hum_front_afternoon_at_3km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_3km(barcis,sappada_forni_villa).
hum_front_afternoon_at_3km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_5_5km(barcis,pordenone).
hum_front_morning_at_5_5km(pordenone,udine_palmanova).
hum_front_afternoon_at_5_5km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_5_5km(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_5_5km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_5_5km(gorizia,lignano_grado).
