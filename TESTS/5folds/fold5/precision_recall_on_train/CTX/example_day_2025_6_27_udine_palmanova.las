location_considered(udine_palmanova). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 6, 27).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_3km_covers(sappada_forni_villa,1032, 11).
cloud_at_3km_covers(pontebba_tarvisio,1033, 11).
cloud_at_3km_covers(pontebba_tarvisio,1027, 12).
cloud_at_3km_covers(pontebba_tarvisio,1027, 14).
cloud_at_3km_covers(pontebba_tarvisio,1027, 15).
cloud_at_3km_covers(pontebba_tarvisio,1048, 17).
cloud_at_5_5km_covers(gorizia,1033, 1).
cloud_at_9km_covers(sappada_forni_villa,1217, 0).
cloud_at_9km_covers(lignano_grado,1217, 0).
cloud_at_9km_covers(barcis,1217, 0).
cloud_at_9km_covers(udine_palmanova,1217, 0).
cloud_at_9km_covers(gorizia,1217, 0).
cloud_at_9km_covers(pordenone,1217, 0).
cloud_at_9km_covers(pontebba_tarvisio,1224, 1).
cloud_at_9km_covers(lignano_grado,1224, 1).
cloud_at_9km_covers(udine_palmanova,1224, 1).
cloud_at_9km_covers(gorizia,1224, 1).
cloud_at_9km_covers(trieste,1224, 1).
cloud_at_9km_covers(gemona_stolvizza,1224, 1).
cloud_at_9km_covers(pontebba_tarvisio,1224, 2).
cloud_at_9km_covers(trieste,1224, 2).
cloud_at_9km_covers(sappada_forni_villa,1240, 14).
cloud_at_9km_covers(barcis,1240, 14).
cloud_at_9km_covers(pordenone,1240, 14).
cloud_at_9km_covers(sappada_forni_villa,1240, 15).
cloud_at_9km_covers(pontebba_tarvisio,1240, 15).
cloud_at_9km_covers(lignano_grado,1240, 15).
cloud_at_9km_covers(barcis,1240, 15).
cloud_at_9km_covers(udine_palmanova,1240, 15).
cloud_at_9km_covers(gorizia,1240, 15).
cloud_at_9km_covers(gemona_stolvizza,1240, 15).
cloud_at_9km_covers(sappada_forni_villa,1240, 16).
cloud_at_9km_covers(lignano_grado,1240, 16).
cloud_at_9km_covers(barcis,1240, 16).
cloud_at_9km_covers(gorizia,1240, 16).
cloud_at_9km_covers(trieste,1240, 16).
cloud_at_9km_covers(pordenone,1240, 16).
cloud_at_9km_covers(lignano_grado,1250, 17).
cloud_at_9km_covers(udine_palmanova,1250, 17).
cloud_at_9km_covers(trieste,1250, 17).
cloud_at_9km_covers(sappada_forni_villa,1256, 20).
cloud_at_9km_covers(barcis,1256, 20).
cloud_at_9km_covers(barcis,1256, 21).
cloud_at_9km_covers(pontebba_tarvisio,1256, 22).
cloud_at_9km_covers(lignano_grado,1256, 22).
cloud_at_9km_covers(udine_palmanova,1256, 22).
cloud_at_9km_covers(gorizia,1256, 22).
cloud_at_9km_covers(gemona_stolvizza,1256, 22).
cloud_at_9km_covers(pordenone,1256, 22).
cloud_at_9km_covers(lignano_grado,1262, 23).
cloud_at_9km_covers(barcis,1262, 23).
cloud_at_9km_covers(gemona_stolvizza,1262, 23).
cloud_at_9km_covers(pordenone,1262, 23).
cloud_at_9km_covers(trieste,1256, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,273.73).
% temperature_at_afternoon(sappada_forni_villa,275.71).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,277.63).
% temperature_at_afternoon(pontebba_tarvisio,274.81).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,274.00).
% temperature_at_afternoon(lignano_grado,274.23).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,272.90).
% temperature_at_afternoon(barcis,275.50).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,274.03).
% temperature_at_afternoon(udine_palmanova,273.81).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,273.93).
% temperature_at_afternoon(gorizia,274.06).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,274.03).
% temperature_at_afternoon(trieste,274.35).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,274.43).
% temperature_at_afternoon(gemona_stolvizza,275.29).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,272.73).
% temperature_at_afternoon(pordenone,273.42).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,69.33).
% humidity_at_afternoon(sappada_forni_villa,40.83).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,56.67).
% humidity_at_afternoon(pontebba_tarvisio,49.58).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,45.33).
% humidity_at_afternoon(lignano_grado,60.00).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,58.67).
% humidity_at_afternoon(barcis,47.08).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,47.33).
% humidity_at_afternoon(udine_palmanova,45.42).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,50.00).
% humidity_at_afternoon(gorizia,65.00).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,45.33).
% humidity_at_afternoon(trieste,60.00).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,64.67).
% humidity_at_afternoon(gemona_stolvizza,51.67).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,46.67).
% humidity_at_afternoon(pordenone,47.92).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",17).
wind_blowing_afternoon(sappada_forni_villa,"S",18).
wind_blowing_morning(pontebba_tarvisio,"S",14).
wind_blowing_afternoon(pontebba_tarvisio,"SE",18).
wind_blowing_morning(lignano_grado,"S",17).
wind_blowing_afternoon(lignano_grado,"SE",17).
wind_blowing_morning(barcis,"S",17).
wind_blowing_afternoon(barcis,"S",18).
wind_blowing_morning(udine_palmanova,"S",14).
wind_blowing_afternoon(udine_palmanova,"SE",16).
wind_blowing_morning(gorizia,"S",14).
wind_blowing_afternoon(gorizia,"SE",16).
wind_blowing_morning(trieste,"S",14).
wind_blowing_afternoon(trieste,"SE",16).
wind_blowing_morning(gemona_stolvizza,"S",14).
wind_blowing_afternoon(gemona_stolvizza,"SE",18).
wind_blowing_morning(pordenone,"S",17).
wind_blowing_afternoon(pordenone,"SE",17).

temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_afternoon_at_3km(barcis,sappada_forni_villa).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
hum_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(lignano_grado,trieste).
hum_front_afternoon_at_1_4km(gorizia,trieste).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
