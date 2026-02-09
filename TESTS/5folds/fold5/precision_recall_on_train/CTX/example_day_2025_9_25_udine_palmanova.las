location_considered(udine_palmanova). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 9, 25).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(barcis,1396, 22).
cloud_at_3km_covers(lignano_grado,1664, 22).
cloud_at_3km_covers(udine_palmanova,1664, 22).
cloud_at_3km_covers(udine_palmanova,1664, 23).
cloud_at_3km_covers(gorizia,1664, 23).
cloud_at_3km_covers(gemona_stolvizza,1664, 23).
cloud_at_5_5km_covers(sappada_forni_villa,1630, 17).
cloud_at_5_5km_covers(barcis,1630, 17).
cloud_at_5_5km_covers(sappada_forni_villa,1635, 21).
cloud_at_9km_covers(pontebba_tarvisio,1948, 22).
cloud_at_9km_covers(lignano_grado,1950, 22).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,277.70).
% temperature_at_afternoon(sappada_forni_villa,277.79).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,277.37).
% temperature_at_afternoon(pontebba_tarvisio,277.21).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,276.20).
% temperature_at_afternoon(lignano_grado,274.69).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,278.33).
% temperature_at_afternoon(barcis,278.58).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.87).
% temperature_at_afternoon(udine_palmanova,275.21).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,275.40).
% temperature_at_afternoon(gorizia,274.42).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,273.87).
% temperature_at_afternoon(trieste,274.10).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,277.77).
% temperature_at_afternoon(gemona_stolvizza,276.50).
temperature_decreased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,278.03).
% temperature_at_afternoon(pordenone,275.96).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,52.00).
% humidity_at_afternoon(sappada_forni_villa,51.67).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,42.67).
% humidity_at_afternoon(pontebba_tarvisio,55.42).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,61.33).
% humidity_at_afternoon(lignano_grado,54.58).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,56.00).
% humidity_at_afternoon(barcis,51.67).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,70.67).
% humidity_at_afternoon(udine_palmanova,63.33).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,68.67).
% humidity_at_afternoon(gorizia,67.92).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,73.33).
% humidity_at_afternoon(trieste,60.00).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,56.67).
% humidity_at_afternoon(gemona_stolvizza,67.50).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,64.00).
% humidity_at_afternoon(pordenone,50.83).
humidity_decreased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"NE",36).
wind_blowing_afternoon(sappada_forni_villa,"NE",34).
wind_blowing_morning(pontebba_tarvisio,"NE",33).
wind_blowing_afternoon(pontebba_tarvisio,"NE",31).
wind_blowing_morning(lignano_grado,"NE",36).
wind_blowing_afternoon(lignano_grado,"NE",34).
wind_blowing_morning(barcis,"NE",36).
wind_blowing_afternoon(barcis,"NE",34).
wind_blowing_morning(udine_palmanova,"NE",33).
wind_blowing_afternoon(udine_palmanova,"NE",31).
wind_blowing_morning(gorizia,"NE",33).
wind_blowing_afternoon(gorizia,"NE",31).
wind_blowing_morning(trieste,"NE",33).
wind_blowing_afternoon(trieste,"NE",31).
wind_blowing_morning(gemona_stolvizza,"NE",33).
wind_blowing_afternoon(gemona_stolvizza,"NE",31).
wind_blowing_morning(pordenone,"NE",36).
wind_blowing_afternoon(pordenone,"NE",34).
