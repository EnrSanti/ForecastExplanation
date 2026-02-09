location_considered(pordenone). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 1, 29).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_100m_covers(gemona_stolvizza,194, 0).
cloud_at_100m_covers(gemona_stolvizza,194, 1).
cloud_at_100m_covers(pordenone,194, 1).
cloud_at_100m_covers(pordenone,196, 2).
cloud_at_100m_covers(pordenone,196, 3).
cloud_at_100m_covers(pordenone,196, 4).
cloud_at_750m_covers(gemona_stolvizza,183, 0).
cloud_at_750m_covers(gemona_stolvizza,183, 1).
cloud_at_1_4km_covers(gemona_stolvizza,236, 0).
cloud_at_1_4km_covers(trieste,234, 1).
cloud_at_1_4km_covers(trieste,238, 2).
cloud_at_1_4km_covers(trieste,242, 10).
cloud_at_1_4km_covers(gemona_stolvizza,243, 12).
cloud_at_1_4km_covers(sappada_forni_villa,247, 18).
cloud_at_1_4km_covers(gemona_stolvizza,247, 18).
cloud_at_1_4km_covers(barcis,247, 19).
cloud_at_1_4km_covers(barcis,247, 20).
cloud_at_3km_covers(pontebba_tarvisio,212, 0).
cloud_at_3km_covers(gemona_stolvizza,212, 0).
cloud_at_3km_covers(pontebba_tarvisio,212, 1).
cloud_at_3km_covers(udine_palmanova,231, 18).
cloud_at_5_5km_covers(sappada_forni_villa,324, 4).
cloud_at_9km_covers(barcis,351, 18).
cloud_at_9km_covers(pontebba_tarvisio,350, 19).
cloud_at_9km_covers(lignano_grado,350, 19).
cloud_at_9km_covers(barcis,350, 19).
cloud_at_9km_covers(udine_palmanova,350, 19).
cloud_at_9km_covers(gorizia,350, 19).
cloud_at_9km_covers(gemona_stolvizza,350, 19).
cloud_at_9km_covers(pontebba_tarvisio,354, 20).
cloud_at_9km_covers(sappada_forni_villa,356, 22).
cloud_at_9km_covers(sappada_forni_villa,356, 23).
cloud_at_9km_covers(pontebba_tarvisio,356, 23).
cloud_at_9km_covers(gemona_stolvizza,356, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,279.13).
% temperature_at_afternoon(sappada_forni_villa,278.90).
temperature_decreased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.87).
% temperature_at_afternoon(pontebba_tarvisio,278.83).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.27).
% temperature_at_afternoon(lignano_grado,276.85).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,278.53).
% temperature_at_afternoon(barcis,278.92).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,278.40).
% temperature_at_afternoon(udine_palmanova,277.38).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,277.77).
% temperature_at_afternoon(gorizia,277.17).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,275.93).
% temperature_at_afternoon(trieste,275.50).
temperature_decreased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,279.00).
% temperature_at_afternoon(gemona_stolvizza,278.96).
temperature_decreased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,277.60).
% temperature_at_afternoon(pordenone,277.65).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,43.33).
% humidity_at_afternoon(sappada_forni_villa,39.17).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,33.33).
% humidity_at_afternoon(pontebba_tarvisio,31.67).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,46.00).
% humidity_at_afternoon(lignano_grado,45.83).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,56.00).
% humidity_at_afternoon(barcis,35.42).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,48.00).
% humidity_at_afternoon(udine_palmanova,37.92).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,38.00).
% humidity_at_afternoon(gorizia,37.08).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,48.67).
% humidity_at_afternoon(trieste,44.58).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,43.33).
% humidity_at_afternoon(gemona_stolvizza,32.08).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,58.67).
% humidity_at_afternoon(pordenone,37.08).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"NE",18).
wind_blowing_afternoon(sappada_forni_villa,"SE",10).
wind_blowing_morning(pontebba_tarvisio,"NE",16).
wind_blowing_afternoon(pontebba_tarvisio,"E",7).
wind_blowing_morning(lignano_grado,"NE",16).
wind_blowing_afternoon(lignano_grado,"SE",8).
wind_blowing_morning(barcis,"NE",18).
wind_blowing_afternoon(barcis,"SE",10).
wind_blowing_morning(udine_palmanova,"NE",16).
wind_blowing_afternoon(udine_palmanova,"E",7).
wind_blowing_morning(gorizia,"NE",16).
wind_blowing_afternoon(gorizia,"E",7).
wind_blowing_morning(trieste,"NE",16).
wind_blowing_afternoon(trieste,"E",7).
wind_blowing_morning(gemona_stolvizza,"NE",16).
wind_blowing_afternoon(gemona_stolvizza,"E",7).
wind_blowing_morning(pordenone,"NE",16).
wind_blowing_afternoon(pordenone,"SE",8).

temp_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_morning_at_750m(lignano_grado,udine_palmanova).
temp_front_morning_at_750m(gorizia,lignano_grado).
temp_front_afternoon_at_750m(lignano_grado,trieste).
temp_front_afternoon_at_750m(gorizia,trieste).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_1_4km(barcis,pordenone).
temp_front_morning_at_1_4km(gorizia,lignano_grado).
temp_front_afternoon_at_3km(gorizia,lignano_grado).
temp_front_morning_at_5_5km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_5_5km(sappada_forni_villa,udine_palmanova).

hum_front_morning_at_1_4km(lignano_grado,pordenone).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_morning_at_3km(gemona_stolvizza,gorizia).
hum_front_morning_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_3km(pordenone,udine_palmanova).
hum_front_afternoon_at_3km(lignano_grado,pordenone).
hum_front_afternoon_at_3km(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_5_5km(barcis,sappada_forni_villa).
hum_front_afternoon_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
hum_front_morning_at_9km(gorizia,pontebba_tarvisio).
