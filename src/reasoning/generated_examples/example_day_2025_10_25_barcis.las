% Example generated data for day (2025, 10, 25)

#pos(e216@1000,{ 

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
date(2025, 10, 25).

% Cloud coverage data:
% Cloud_covers(location,cloud_id,hh)
cloud_at_1_4km_covers(pordenone,1703, 21).
cloud_at_3km_covers(gorizia,1927, 12).
cloud_at_3km_covers(gorizia,1927, 13).
cloud_at_5_5km_covers(pontebba_tarvisio,1951, 1).
cloud_at_5_5km_covers(lignano_grado,1960, 8).
cloud_at_5_5km_covers(pordenone,1960, 8).
cloud_at_5_5km_covers(gorizia,1964, 12).
cloud_at_5_5km_covers(gorizia,1972, 13).
cloud_at_5_5km_covers(sappada_forni_villa,1975, 18).
cloud_at_5_5km_covers(sappada_forni_villa,1980, 21).
cloud_at_5_5km_covers(sappada_forni_villa,1980, 22).
cloud_at_5_5km_covers(pontebba_tarvisio,1980, 22).
cloud_at_5_5km_covers(gemona_stolvizza,1980, 22).
cloud_at_5_5km_covers(pordenone,1981, 22).
cloud_at_5_5km_covers(udine_palmanova,1981, 23).
cloud_at_5_5km_covers(pordenone,1981, 23).
cloud_at_9km_covers(sappada_forni_villa,2338, 1).
cloud_at_9km_covers(pontebba_tarvisio,2338, 1).
cloud_at_9km_covers(lignano_grado,2338, 1).
cloud_at_9km_covers(barcis,2338, 1).
cloud_at_9km_covers(udine_palmanova,2338, 1).
cloud_at_9km_covers(gorizia,2338, 1).
cloud_at_9km_covers(gemona_stolvizza,2338, 1).
cloud_at_9km_covers(pordenone,2338, 1).
cloud_at_9km_covers(sappada_forni_villa,2341, 2).
cloud_at_9km_covers(pontebba_tarvisio,2341, 2).
cloud_at_9km_covers(lignano_grado,2341, 2).
cloud_at_9km_covers(barcis,2341, 2).
cloud_at_9km_covers(udine_palmanova,2341, 2).
cloud_at_9km_covers(gorizia,2341, 2).
cloud_at_9km_covers(gemona_stolvizza,2341, 2).
cloud_at_9km_covers(pordenone,2341, 2).
cloud_at_9km_covers(pontebba_tarvisio,2338, 3).
cloud_at_9km_covers(lignano_grado,2338, 3).
cloud_at_9km_covers(barcis,2338, 3).
cloud_at_9km_covers(udine_palmanova,2338, 3).
cloud_at_9km_covers(gorizia,2338, 3).
cloud_at_9km_covers(trieste,2338, 3).
cloud_at_9km_covers(gemona_stolvizza,2338, 3).
cloud_at_9km_covers(pordenone,2338, 3).
cloud_at_9km_covers(pontebba_tarvisio,2338, 4).
cloud_at_9km_covers(lignano_grado,2338, 4).
cloud_at_9km_covers(udine_palmanova,2338, 4).
cloud_at_9km_covers(gorizia,2338, 4).
cloud_at_9km_covers(trieste,2338, 4).
cloud_at_9km_covers(gemona_stolvizza,2338, 4).
cloud_at_9km_covers(pordenone,2338, 4).
cloud_at_9km_covers(barcis,2342, 5).
cloud_at_9km_covers(udine_palmanova,2342, 5).
cloud_at_9km_covers(gorizia,2342, 5).
cloud_at_9km_covers(trieste,2342, 5).
cloud_at_9km_covers(gemona_stolvizza,2342, 5).
cloud_at_9km_covers(pordenone,2342, 5).
cloud_at_9km_covers(lignano_grado,2350, 19).
cloud_at_9km_covers(trieste,2350, 19).
cloud_at_9km_covers(trieste,2350, 20).

%summing up temperature and humidity facts 

% temperature_at_morning(sappada_forni_villa,279.00).
% temperature_at_afternoon(sappada_forni_villa,279.08).
temperature_increased_at_afternoon(sappada_forni_villa).
% temperature_at_morning(pontebba_tarvisio,278.97).
% temperature_at_afternoon(pontebba_tarvisio,279.06).
temperature_increased_at_afternoon(pontebba_tarvisio).
% temperature_at_morning(lignano_grado,274.77).
% temperature_at_afternoon(lignano_grado,273.50).
temperature_decreased_at_afternoon(lignano_grado).
% temperature_at_morning(barcis,278.07).
% temperature_at_afternoon(barcis,278.23).
temperature_increased_at_afternoon(barcis).
% temperature_at_morning(udine_palmanova,276.97).
% temperature_at_afternoon(udine_palmanova,275.12).
temperature_decreased_at_afternoon(udine_palmanova).
% temperature_at_morning(gorizia,275.83).
% temperature_at_afternoon(gorizia,274.81).
temperature_decreased_at_afternoon(gorizia).
% temperature_at_morning(trieste,273.47).
% temperature_at_afternoon(trieste,273.77).
temperature_increased_at_afternoon(trieste).
% temperature_at_morning(gemona_stolvizza,278.30).
% temperature_at_afternoon(gemona_stolvizza,278.65).
temperature_increased_at_afternoon(gemona_stolvizza).
% temperature_at_morning(pordenone,276.57).
% temperature_at_afternoon(pordenone,275.48).
temperature_decreased_at_afternoon(pordenone).
% humidity_at_morning(sappada_forni_villa,62.67).
% humidity_at_afternoon(sappada_forni_villa,50.00).
humidity_decreased_at_afternoon(sappada_forni_villa).
% humidity_at_morning(pontebba_tarvisio,44.67).
% humidity_at_afternoon(pontebba_tarvisio,55.42).
humidity_increased_at_afternoon(pontebba_tarvisio).
% humidity_at_morning(lignano_grado,51.33).
% humidity_at_afternoon(lignano_grado,57.08).
humidity_increased_at_afternoon(lignano_grado).
% humidity_at_morning(barcis,57.33).
% humidity_at_afternoon(barcis,60.00).
humidity_increased_at_afternoon(barcis).
% humidity_at_morning(udine_palmanova,48.67).
% humidity_at_afternoon(udine_palmanova,56.25).
humidity_increased_at_afternoon(udine_palmanova).
% humidity_at_morning(gorizia,46.67).
% humidity_at_afternoon(gorizia,54.58).
humidity_increased_at_afternoon(gorizia).
% humidity_at_morning(trieste,48.67).
% humidity_at_afternoon(trieste,65.83).
humidity_increased_at_afternoon(trieste).
% humidity_at_morning(gemona_stolvizza,43.33).
% humidity_at_afternoon(gemona_stolvizza,57.08).
humidity_increased_at_afternoon(gemona_stolvizza).
% humidity_at_morning(pordenone,53.33).
% humidity_at_afternoon(pordenone,60.42).
humidity_increased_at_afternoon(pordenone).
wind_blowing_morning(sappada_forni_villa,"E",43).
wind_blowing_afternoon(sappada_forni_villa,"E",55).
wind_blowing_morning(pontebba_tarvisio,"E",45).
wind_blowing_afternoon(pontebba_tarvisio,"E",55).
wind_blowing_morning(lignano_grado,"E",46).
wind_blowing_afternoon(lignano_grado,"E",63).
wind_blowing_morning(barcis,"E",43).
wind_blowing_afternoon(barcis,"E",55).
wind_blowing_morning(udine_palmanova,"E",45).
wind_blowing_afternoon(udine_palmanova,"E",61).
wind_blowing_morning(gorizia,"E",45).
wind_blowing_afternoon(gorizia,"E",61).
wind_blowing_morning(trieste,"E",45).
wind_blowing_afternoon(trieste,"E",61).
wind_blowing_morning(gemona_stolvizza,"E",45).
wind_blowing_afternoon(gemona_stolvizza,"E",55).
wind_blowing_morning(pordenone,"E",46).
wind_blowing_afternoon(pordenone,"E",63).

temp_front_morning_at_100m(gorizia,lignano_grado).
temp_front_afternoon_at_100m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_100m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_100m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_100m(pordenone,sappada_forni_villa).
temp_front_afternoon_at_100m(barcis,sappada_forni_villa).
temp_front_morning_at_750m(gorizia,pontebba_tarvisio).
temp_front_morning_at_750m(barcis,pordenone).
temp_front_morning_at_750m(lignano_grado,udine_palmanova).
temp_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_750m(barcis,pordenone).
temp_front_afternoon_at_750m(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_750m(pordenone,sappada_forni_villa).
temp_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
temp_front_morning_at_1_4km(barcis,pordenone).
temp_front_morning_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_morning_at_1_4km(gemona_stolvizza,udine_palmanova).
temp_front_morning_at_1_4km(pordenone,sappada_forni_villa).
temp_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
temp_front_afternoon_at_1_4km(gorizia,pontebba_tarvisio).
temp_front_afternoon_at_1_4km(gemona_stolvizza,udine_palmanova).
temp_front_afternoon_at_1_4km(pordenone,sappada_forni_villa).
temp_front_morning_at_3km(barcis,sappada_forni_villa).
temp_front_afternoon_at_9km(gorizia,pontebba_tarvisio).

hum_front_morning_at_100m(lignano_grado,pordenone).
hum_front_morning_at_100m(pordenone,udine_palmanova).
hum_front_afternoon_at_100m(barcis,pordenone).
hum_front_afternoon_at_100m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(barcis,pordenone).
hum_front_afternoon_at_750m(gorizia,lignano_grado).
hum_front_afternoon_at_750m(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_750m(gemona_stolvizza,udine_palmanova).
hum_front_afternoon_at_750m(pordenone,sappada_forni_villa).
hum_front_morning_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_1_4km(barcis,pordenone).
hum_front_morning_at_1_4km(gemona_stolvizza,udine_palmanova).
hum_front_morning_at_1_4km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_1_4km(sappada_forni_villa,udine_palmanova).
hum_front_afternoon_at_1_4km(pordenone,udine_palmanova).
hum_front_morning_at_3km(sappada_forni_villa,udine_palmanova).
hum_front_morning_at_3km(pordenone,sappada_forni_villa).
hum_front_afternoon_at_3km(pordenone,udine_palmanova).

}). 
