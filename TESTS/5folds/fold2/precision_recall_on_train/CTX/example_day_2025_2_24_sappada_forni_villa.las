location_considered(sappada_forni_villa). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 2, 24).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_750m_covers(trieste,194, 0).
cloud_at_1_4km_covers(sappada_forni_villa,317, 17).
cloud_at_1_4km_covers(sappada_forni_villa,317, 18).
cloud_at_3km_covers(pontebba_tarvisio,278, 6).
cloud_at_3km_covers(gemona_stolvizza,279, 6).
cloud_at_3km_covers(gorizia,282, 7).
cloud_at_3km_covers(trieste,282, 7).
cloud_at_3km_covers(trieste,284, 8).
cloud_at_3km_covers(trieste,279, 9).
cloud_at_3km_covers(trieste,287, 10).
cloud_at_5_5km_covers(sappada_forni_villa,391, 0).
cloud_at_5_5km_covers(pontebba_tarvisio,391, 0).
cloud_at_5_5km_covers(pontebba_tarvisio,391, 1).
cloud_at_5_5km_covers(pordenone,406, 1).
cloud_at_5_5km_covers(pontebba_tarvisio,391, 2).
cloud_at_5_5km_covers(pordenone,406, 2).
cloud_at_5_5km_covers(pontebba_tarvisio,414, 3).
cloud_at_5_5km_covers(pordenone,406, 3).
cloud_at_5_5km_covers(pontebba_tarvisio,414, 4).
cloud_at_5_5km_covers(lignano_grado,414, 4).
cloud_at_5_5km_covers(udine_palmanova,414, 4).
cloud_at_5_5km_covers(gorizia,414, 4).
cloud_at_5_5km_covers(trieste,414, 4).
cloud_at_5_5km_covers(lignano_grado,414, 5).
cloud_at_5_5km_covers(gorizia,414, 5).
cloud_at_5_5km_covers(trieste,414, 5).
cloud_at_5_5km_covers(trieste,414, 6).
cloud_at_5_5km_covers(trieste,414, 7).
cloud_at_9km_covers(barcis,500, 8).
cloud_at_9km_covers(pontebba_tarvisio,503, 9).
cloud_at_9km_covers(udine_palmanova,503, 9).
cloud_at_9km_covers(gemona_stolvizza,503, 9).
cloud_at_9km_covers(pordenone,504, 9).
cloud_at_9km_covers(lignano_grado,505, 10).
cloud_at_9km_covers(gorizia,505, 10).
cloud_at_9km_covers(trieste,505, 10).
cloud_at_9km_covers(sappada_forni_villa,506, 17).
cloud_at_9km_covers(barcis,506, 17).
cloud_at_9km_covers(pordenone,506, 17).
cloud_at_9km_covers(lignano_grado,506, 18).
cloud_at_9km_covers(udine_palmanova,506, 18).
cloud_at_9km_covers(gorizia,506, 18).
cloud_at_9km_covers(gemona_stolvizza,506, 18).
cloud_at_9km_covers(pontebba_tarvisio,506, 19).
cloud_at_9km_covers(gorizia,506, 19).
cloud_at_9km_covers(trieste,506, 19).
cloud_at_9km_covers(pordenone,506, 19).
cloud_at_9km_covers(barcis,506, 20).
cloud_at_9km_covers(pordenone,506, 20).
cloud_at_9km_covers(sappada_forni_villa,506, 21).
cloud_at_9km_covers(lignano_grado,506, 21).
cloud_at_9km_covers(barcis,506, 21).
cloud_at_9km_covers(gemona_stolvizza,506, 21).
cloud_at_9km_covers(pordenone,506, 21).
cloud_at_9km_covers(sappada_forni_villa,515, 22).
cloud_at_9km_covers(barcis,515, 22).
cloud_at_9km_covers(pordenone,515, 22).
cloud_at_9km_covers(sappada_forni_villa,511, 23).
cloud_at_9km_covers(barcis,511, 23).
cloud_at_9km_covers(trieste,519, 23).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,277.40).
% temperature_at_afternoon(sappada_forni_villa,277.92).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,277.77).
% temperature_at_afternoon(pontebba_tarvisio,277.81).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,278.77).
% temperature_at_afternoon(lignano_grado,278.94).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,276.20).
% temperature_at_afternoon(barcis,277.90).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,278.60).
% temperature_at_afternoon(udine_palmanova,278.23).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,278.23).
% temperature_at_afternoon(gorizia,278.52).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,277.83).
% temperature_at_afternoon(trieste,279.29).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.73).
% temperature_at_afternoon(gemona_stolvizza,278.19).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,278.63).
% temperature_at_afternoon(pordenone,278.50).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,39.33).
% humidity_at_afternoon(sappada_forni_villa,31.67).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,34.00).
% humidity_at_afternoon(pontebba_tarvisio,29.17).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,42.67).
% humidity_at_afternoon(lignano_grado,45.42).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,44.67).
% humidity_at_afternoon(barcis,32.50).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,54.67).
% humidity_at_afternoon(udine_palmanova,43.75).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,47.33).
% humidity_at_afternoon(gorizia,39.17).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,47.33).
% humidity_at_afternoon(trieste,43.75).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,48.00).
% humidity_at_afternoon(gemona_stolvizza,32.50).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,46.67).
% humidity_at_afternoon(pordenone,48.75).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",30).
wind_blowing_afternoon(sappada_forni_villa,"S",20).
wind_blowing_morning(pontebba_tarvisio,"S",35).
wind_blowing_afternoon(pontebba_tarvisio,"S",23).
wind_blowing_morning(lignano_grado,"S",31).
wind_blowing_afternoon(lignano_grado,"S",20).
wind_blowing_morning(barcis,"S",30).
wind_blowing_afternoon(barcis,"S",20).
wind_blowing_morning(udine_palmanova,"S",35).
wind_blowing_afternoon(udine_palmanova,"S",24).
wind_blowing_morning(gorizia,"S",35).
wind_blowing_afternoon(gorizia,"S",24).
wind_blowing_morning(trieste,"S",35).
wind_blowing_afternoon(trieste,"S",24).
wind_blowing_morning(gemona_stolvizza,"S",35).
wind_blowing_afternoon(gemona_stolvizza,"S",23).
wind_blowing_morning(pordenone,"S",31).
wind_blowing_afternoon(pordenone,"S",20).

temp_front_morning_at_100m(barcis,pordenone).
temp_front_afternoon_at_100m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_afternoon_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_5_5km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_5_5km(lignano_grado,pordenone).
temp_front_morning_at_5_5km(gemona_stolvizza,sappada_forni_villa).
temp_front_morning_at_5_5km(pordenone,udine_palmanova).

hum_front_morning_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_100m(lignano_grado,udine_palmanova).
hum_front_morning_at_100m(gorizia,lignano_grado).
hum_front_morning_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(lignano_grado,udine_palmanova).
hum_front_morning_at_750m(pordenone,udine_palmanova).
hum_front_morning_at_750m(gorizia,lignano_grado).
hum_front_morning_at_5_5km(pontebba_tarvisio,sappada_forni_villa).
