location_considered(pontebba_tarvisio). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 6).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_9km_covers(sappada_forni_villa,1665, 12).
cloud_at_9km_covers(barcis,1665, 12).
cloud_at_9km_covers(pordenone,1665, 12).
cloud_at_9km_covers(sappada_forni_villa,1669, 13).
cloud_at_9km_covers(barcis,1669, 13).
cloud_at_9km_covers(pordenone,1669, 13).
cloud_at_9km_covers(sappada_forni_villa,1669, 14).
cloud_at_9km_covers(lignano_grado,1669, 14).
cloud_at_9km_covers(barcis,1669, 14).
cloud_at_9km_covers(gemona_stolvizza,1669, 14).
cloud_at_9km_covers(pordenone,1669, 14).
cloud_at_9km_covers(lignano_grado,1669, 15).
cloud_at_9km_covers(udine_palmanova,1669, 15).
cloud_at_9km_covers(gorizia,1669, 15).
cloud_at_9km_covers(gemona_stolvizza,1669, 15).
cloud_at_9km_covers(trieste,1672, 16).
cloud_at_9km_covers(sappada_forni_villa,1673, 22).
cloud_at_9km_covers(pontebba_tarvisio,1673, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,274.60).
% temperature_at_afternoon(sappada_forni_villa,276.10).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.63).
% temperature_at_afternoon(pontebba_tarvisio,277.19).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,273.47).
% temperature_at_afternoon(lignano_grado,274.96).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,273.17).
% temperature_at_afternoon(barcis,277.04).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,273.20).
% temperature_at_afternoon(udine_palmanova,275.75).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,273.37).
% temperature_at_afternoon(gorizia,276.46).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,272.93).
% temperature_at_afternoon(trieste,275.00).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,273.47).
% temperature_at_afternoon(gemona_stolvizza,277.00).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,273.40).
% temperature_at_afternoon(pordenone,274.58).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,72.67).
% humidity_at_afternoon(sappada_forni_villa,50.42).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,49.33).
% humidity_at_afternoon(pontebba_tarvisio,45.42).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,59.33).
% humidity_at_afternoon(lignano_grado,49.58).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,51.33).
% humidity_at_afternoon(barcis,43.33).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,66.00).
% humidity_at_afternoon(udine_palmanova,52.92).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,74.67).
% humidity_at_afternoon(gorizia,58.75).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,51.33).
% humidity_at_afternoon(trieste,52.50).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,59.33).
% humidity_at_afternoon(gemona_stolvizza,49.58).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,29.33).
% humidity_at_afternoon(pordenone,64.17).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"SE",12).
wind_blowing_afternoon(sappada_forni_villa,"SE",19).
wind_blowing_morning(pontebba_tarvisio,"SE",11).
wind_blowing_afternoon(pontebba_tarvisio,"SE",18).
wind_blowing_morning(lignano_grado,"SE",12).
wind_blowing_afternoon(lignano_grado,"SE",18).
wind_blowing_morning(barcis,"SE",12).
wind_blowing_afternoon(barcis,"SE",19).
wind_blowing_morning(udine_palmanova,"SE",9).
wind_blowing_afternoon(udine_palmanova,"SE",17).
wind_blowing_morning(gorizia,"SE",9).
wind_blowing_afternoon(gorizia,"SE",17).
wind_blowing_morning(trieste,"SE",9).
wind_blowing_afternoon(trieste,"SE",17).
wind_blowing_morning(gemona_stolvizza,"SE",11).
wind_blowing_afternoon(gemona_stolvizza,"SE",18).
wind_blowing_morning(pordenone,"SE",12).
wind_blowing_afternoon(pordenone,"SE",18).

temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_100m(barcis,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_3km(gorizia,pontebba_tarvisio).
temp_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_3km(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_5_5km(pordenone,sappada_forni_villa).
temp_front_afternoon_at_9km(lignano_grado,pordenone).
temp_front_afternoon_at_9km(pordenone,udine_palmanova).
temp_front_afternoon_at_9km(pontebba_tarvisio,sappada_forni_villa).

hum_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_100m(lignano_grado,trieste).
hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_100m(lignano_grado,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_afternoon_at_100m(pordenone,udine_palmanova).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_750m(lignano_grado,udine_palmanova).
hum_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_1_4km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_1_4km(gemona_stolvizza,sappada_forni_villa).
hum_front_morning_at_1_4km(pordenone,sappada_forni_villa).
hum_front_morning_at_1_4km(barcis,sappada_forni_villa).
hum_front_morning_at_3km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_morning_at_3km(lignano_grado,trieste).
hum_front_morning_at_3km(gemona_stolvizza,pontebba_tarvisio).
hum_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_3km(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_5_5km(lignano_grado,udine_palmanova).
hum_front_morning_at_5_5km(gorizia,lignano_grado).
hum_front_morning_at_9km(gemona_stolvizza,gorizia).
