location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 5, 10).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_100m_covers(pontebba_tarvisio,648, 1).
cloud_at_100m_covers(pontebba_tarvisio,648, 2).
cloud_at_750m_covers(pontebba_tarvisio,633, 1).
cloud_at_750m_covers(pontebba_tarvisio,633, 2).
cloud_at_1_4km_covers(pontebba_tarvisio,874, 0).
cloud_at_1_4km_covers(pontebba_tarvisio,874, 1).
cloud_at_1_4km_covers(pontebba_tarvisio,874, 2).
cloud_at_1_4km_covers(pontebba_tarvisio,874, 3).
cloud_at_3km_covers(pontebba_tarvisio,968, 5).
cloud_at_3km_covers(pontebba_tarvisio,968, 6).
cloud_at_3km_covers(gemona_stolvizza,968, 6).
cloud_at_5_5km_covers(sappada_forni_villa,968, 0).
cloud_at_5_5km_covers(pontebba_tarvisio,968, 0).
cloud_at_5_5km_covers(gemona_stolvizza,968, 0).
cloud_at_5_5km_covers(sappada_forni_villa,968, 1).
cloud_at_5_5km_covers(pontebba_tarvisio,968, 1).
cloud_at_5_5km_covers(barcis,968, 1).
cloud_at_5_5km_covers(gemona_stolvizza,968, 1).
cloud_at_5_5km_covers(barcis,979, 2).
cloud_at_5_5km_covers(gemona_stolvizza,978, 2).
cloud_at_5_5km_covers(gemona_stolvizza,978, 3).
cloud_at_5_5km_covers(udine_palmanova,978, 4).
cloud_at_5_5km_covers(gorizia,978, 4).
cloud_at_5_5km_covers(gemona_stolvizza,978, 4).
cloud_at_9km_covers(trieste,1127, 0).
cloud_at_9km_covers(trieste,1127, 2).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,277.93).
% temperature_at_afternoon(sappada_forni_villa,278.65).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.63).
% temperature_at_afternoon(pontebba_tarvisio,277.33).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.60).
% temperature_at_afternoon(lignano_grado,276.04).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,278.10).
% temperature_at_afternoon(barcis,278.40).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,275.97).
% temperature_at_afternoon(udine_palmanova,275.12).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,275.30).
% temperature_at_afternoon(gorizia,274.65).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,275.07).
% temperature_at_afternoon(trieste,277.69).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.13).
% temperature_at_afternoon(gemona_stolvizza,276.40).
temperature_decreased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,277.60).
% temperature_at_afternoon(pordenone,276.38).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,34.67).
% humidity_at_afternoon(sappada_forni_villa,27.50).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,28.00).
% humidity_at_afternoon(pontebba_tarvisio,40.00).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,47.33).
% humidity_at_afternoon(lignano_grado,45.00).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,34.67).
% humidity_at_afternoon(barcis,32.92).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,32.67).
% humidity_at_afternoon(udine_palmanova,46.25).
humidity_increased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,38.00).
% humidity_at_afternoon(gorizia,42.50).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,46.67).
% humidity_at_afternoon(trieste,52.50).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,56.67).
% humidity_at_afternoon(gemona_stolvizza,60.42).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,47.33).
% humidity_at_afternoon(pordenone,65.42).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"SW",4).
wind_blowing_afternoon(sappada_forni_villa,"S",3).
wind_blowing_morning(pontebba_tarvisio,"SW",4).
wind_blowing_afternoon(pontebba_tarvisio,"S",3).
wind_blowing_morning(lignano_grado,"S",2).
wind_blowing_afternoon(lignano_grado,"S",3).
wind_blowing_morning(barcis,"SW",4).
wind_blowing_afternoon(barcis,"S",3).
wind_blowing_morning(udine_palmanova,"SW",2).
wind_blowing_afternoon(udine_palmanova,"S",3).
wind_blowing_morning(gorizia,"SW",2).
wind_blowing_afternoon(gorizia,"S",3).
wind_blowing_morning(trieste,"SW",2).
wind_blowing_afternoon(trieste,"S",3).
wind_blowing_morning(gemona_stolvizza,"SW",4).
wind_blowing_afternoon(gemona_stolvizza,"S",3).
wind_blowing_morning(pordenone,"S",2).
wind_blowing_afternoon(pordenone,"S",3).

temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(barcis,pordenone).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_100m(gemona_stolvizza,sappada_forni_villa).
temp_front_morning_at_750m(lignano_grado,trieste).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(pordenone,udine_palmanova).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(gemona_stolvizza,sappada_forni_villa).
temp_front_afternoon_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_3km(pordenone,udine_palmanova).
temp_front_morning_at_9km(lignano_grado,udine_palmanova).
temp_front_morning_at_9km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_9km(gorizia,pontebba_tarvisio).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(barcis,pordenone).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_750m(barcis,pordenone).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_750m(pordenone,udine_palmanova).
hum_front_afternoon_at_750m(lignano_grado,trieste).
hum_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_1_4km(lignano_grado,pordenone).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
hum_front_afternoon_at_5_5km(barcis,sappada_forni_villa).
hum_front_afternoon_at_5_5km(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_9km(gorizia,pontebba_tarvisio).
