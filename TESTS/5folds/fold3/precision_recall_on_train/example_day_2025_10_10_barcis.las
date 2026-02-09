% Example generated data for day (2025, 10, 10)

#pos(e210@1000,{ 

forecasted_sky(barcis, "mostly_clear", autumn),
forecasted_rain(barcis, 0, autumn)},
{
partially_sunny_at(barcis,autumn), 
covered_at(barcis,autumn), 
rains_at(barcis,1,autumn), 
rains_at(barcis,2,autumn), 
rains_at(barcis,4,autumn), 
rains_at(barcis,6,autumn)
},
{
location_considered(barcis). 
%to drive the season (winter, spring, summer, autumn)
date(2025, 10, 10).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_5_5km_covers(pontebba_tarvisio,1763, 13).
cloud_at_5_5km_covers(sappada_forni_villa,1763, 14).
cloud_at_5_5km_covers(pontebba_tarvisio,1763, 14).
cloud_at_5_5km_covers(barcis,1763, 14).
cloud_at_5_5km_covers(udine_palmanova,1763, 14).
cloud_at_5_5km_covers(gorizia,1763, 14).
cloud_at_5_5km_covers(gemona_stolvizza,1763, 14).
cloud_at_5_5km_covers(pordenone,1763, 14).
cloud_at_5_5km_covers(sappada_forni_villa,1763, 15).
cloud_at_5_5km_covers(pontebba_tarvisio,1763, 15).
cloud_at_5_5km_covers(lignano_grado,1763, 15).
cloud_at_5_5km_covers(barcis,1763, 15).
cloud_at_5_5km_covers(udine_palmanova,1763, 15).
cloud_at_5_5km_covers(gorizia,1763, 15).
cloud_at_5_5km_covers(trieste,1763, 15).
cloud_at_5_5km_covers(gemona_stolvizza,1763, 15).
cloud_at_5_5km_covers(pordenone,1763, 15).
cloud_at_5_5km_covers(sappada_forni_villa,1767, 16).
cloud_at_5_5km_covers(pontebba_tarvisio,1769, 16).
cloud_at_9km_covers(sappada_forni_villa,2083, 10).
cloud_at_9km_covers(pontebba_tarvisio,2083, 10).
cloud_at_9km_covers(barcis,2083, 10).
cloud_at_9km_covers(sappada_forni_villa,2083, 11).
cloud_at_9km_covers(barcis,2083, 11).
cloud_at_9km_covers(gemona_stolvizza,2083, 11).
cloud_at_9km_covers(lignano_grado,2087, 12).
cloud_at_9km_covers(trieste,2087, 12).
cloud_at_9km_covers(barcis,2086, 12).
cloud_at_9km_covers(lignano_grado,2090, 16).
cloud_at_9km_covers(gorizia,2090, 16).
cloud_at_9km_covers(trieste,2090, 16).
cloud_at_9km_covers(lignano_grado,2093, 18).
cloud_at_9km_covers(gorizia,2093, 18).
cloud_at_9km_covers(trieste,2093, 18).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,272.43).
% temperature_at_afternoon(sappada_forni_villa,273.92).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,275.93).
% temperature_at_afternoon(pontebba_tarvisio,275.10).
temperature_decreased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,273.60).
% temperature_at_afternoon(lignano_grado,274.23).
temperature_increased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,272.73).
% temperature_at_afternoon(barcis,273.81).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,273.63).
% temperature_at_afternoon(udine_palmanova,273.88).
temperature_increased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,273.77).
% temperature_at_afternoon(gorizia,274.48).
temperature_increased_at_afternoon(gorizia).
% temperature_at_morning(trieste,274.73).
% temperature_at_afternoon(trieste,275.48).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,273.33).
% temperature_at_afternoon(gemona_stolvizza,274.67).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,272.73).
% temperature_at_afternoon(pordenone,273.75).
temperature_increased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,52.67).
% humidity_at_afternoon(sappada_forni_villa,40.00).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,54.00).
% humidity_at_afternoon(pontebba_tarvisio,37.92).
humidity_decreased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,66.67).
% humidity_at_afternoon(lignano_grado,55.42).
humidity_decreased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,48.67).
% humidity_at_afternoon(barcis,45.42).
humidity_decreased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,62.67).
% humidity_at_afternoon(udine_palmanova,57.50).
humidity_decreased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,85.33).
% humidity_at_afternoon(gorizia,56.25).
humidity_decreased_at_afternoon(gorizia).
% humidity_at_morning(trieste,73.33).
% humidity_at_afternoon(trieste,52.08).
humidity_decreased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,68.67).
% humidity_at_afternoon(gemona_stolvizza,50.83).
humidity_decreased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,48.67).
% humidity_at_afternoon(pordenone,53.75).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"S",15).
wind_blowing_afternoon(sappada_forni_villa,"S",12).
wind_blowing_morning(pontebba_tarvisio,"S",16).
wind_blowing_afternoon(pontebba_tarvisio,"S",14).
wind_blowing_morning(lignano_grado,"SW",15).
wind_blowing_afternoon(lignano_grado,"S",12).
wind_blowing_morning(barcis,"S",15).
wind_blowing_afternoon(barcis,"S",12).
wind_blowing_morning(udine_palmanova,"SW",15).
wind_blowing_afternoon(udine_palmanova,"S",13).
wind_blowing_morning(gorizia,"SW",15).
wind_blowing_afternoon(gorizia,"S",13).
wind_blowing_morning(trieste,"SW",15).
wind_blowing_afternoon(trieste,"S",13).
wind_blowing_morning(gemona_stolvizza,"S",16).
wind_blowing_afternoon(gemona_stolvizza,"S",14).
wind_blowing_morning(pordenone,"SW",15).
wind_blowing_afternoon(pordenone,"S",12).

temp_front_morning_at_100m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_100m(gorizia,pontebba_tarvisio).
temp_front_morning_at_100m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_750m(pontebba_tarvisio,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
temp_front_morning_at_3km(lignano_grado,trieste).
temp_front_morning_at_9km(lignano_grado,trieste).

hum_front_morning_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
hum_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_100m(pordenone,sappada_forni_villa).
hum_front_morning_at_750m(lignano_grado,pordenone).
hum_front_morning_at_750m(gorizia,pontebba_tarvisio).
hum_front_morning_at_750m(gemona_stolvizza,pontebba_tarvisio).
hum_front_morning_at_750m(pordenone,udine_palmanova).
hum_front_morning_at_750m(barcis,sappada_forni_villa).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(lignano_grado,udine_palmanova).
hum_front_morning_at_1_4km(lignano_grado,pordenone).
hum_front_morning_at_1_4km(barcis,sappada_forni_villa).
hum_front_morning_at_1_4km(pordenone,udine_palmanova).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(lignano_grado,udine_palmanova).
hum_front_afternoon_at_1_4km(barcis,pordenone).
hum_front_afternoon_at_1_4km(gorizia,lignano_grado).
hum_front_afternoon_at_3km(pordenone,udine_palmanova).

}). 
